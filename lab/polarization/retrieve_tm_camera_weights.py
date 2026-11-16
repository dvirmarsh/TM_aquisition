"""
Stage A4 of the plan of 12 Nov 2026 (offline, no hardware): TM retrieval from the 12:34 dataset
with polarization weights on the Fourier-plane camera.

Forward model (TM_retrieve_camweights.py, a copy of TM_retrieve.py):
    image camera:    |E_x|^2 + |E_y|^2                     (unchanged, weights (1, 1))
    Fourier camera:  (1 + b) |E_x|^2 + (1 - b) |E_y|^2
Everything else is the original run: data loading and train/test split (load_real_mmf_data,
split 0.95, seed 42), initialization (TMFocusPipeline.initialize_tm_system), optimizer and loss
(run_tm_focus_optimization.optimize_tm_slm, 100 epochs, lr 0.01, batch 512, both planes), and
the scale between the planes (im_factor). Nothing in run_tm_focus_optimization.py or
TM_retrieve.py is modified; the class is swapped in memory only.

    python retrieve_tm_camera_weights.py --check                     # b = 0 reproduces the model of the current checkpoint
    python retrieve_tm_camera_weights.py --b 0.4 --save-dir <folder> # one retrieval; writes stageA_offline/camweights_b+0.40.json

"Held-out correlation" = Pearson correlation between the predicted and the measured intensity
frame of a held-out probe mask (302 masks, never used in the retrieval), full 128 x 128 frame,
averaged over the masks. Also given: the same inside the fiber support (r <= 58.1 px).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _no_hardware(*_a, **_k):
    raise RuntimeError("This offline script does not use the hardware.")


# run_tm_focus_optimization imports create_refs, which loads the SLM, camera and APT libraries.
# A stub keeps this script free of any hardware library.
_stub = types.ModuleType("create_refs")
for _name in ("create_SLM", "create_SLMpy", "create_cmeras", "set_camera_pixel_format"):
    setattr(_stub, _name, _no_hardware)
sys.modules["create_refs"] = _stub

import run_tm_focus_optimization as rtf            # noqa: E402
import TM_retrieve_camweights as trc               # noqa: E402
from utils import load_real_mmf_data               # noqa: E402

OUT_DIR = TM_LIB / "tm_focus_outputs" / "stageA_offline"
CURRENT_CHECKPOINT = TM_LIB / "tm_focus_outputs" / "tm_system_checkpoint.pt"
SUPPORT_RADIUS = 58.1


def load_data(cfg, pipeline):
    et_ratio = cfg.transmission_image_et[0] / cfg.transmission_fourier_et[0]
    bs_split = 0.47736                                                    # as in TMFocusPipeline.run()
    tran_im_factor = pipeline.magnification ** 2 * bs_split / (1 - bs_split) * et_ratio
    return load_real_mmf_data(h5_file_path=cfg.output_h5_path, experiment_group=cfg.output_group,
                              train_test_split_ratio=cfg.train_test_split_ratio, random_seed=cfg.random_seed,
                              im_factor=tran_im_factor)


def build_system(pipeline, b: float):
    """The system of the original run, built from the copied class, with Fourier-camera weights (1 + b, 1 - b)."""
    rtf.TMMeasurementSystem = trc.TMMeasurementSystem                     # in memory only
    np.random.seed(pipeline.cfg.random_seed)
    tm = pipeline.initialize_tm_system()
    tm.fourier_pol_weights = (1.0 + float(b), 1.0 - float(b))
    return tm


def _pcc_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-30)


def held_out(tm, test_data, chunk: int = 32) -> dict:
    """Correlation between predicted and measured held-out frames, per plane."""
    slm = test_data["slm_patterns"].to(tm.device)
    target = test_data["intensity_targets"].to(tm.device)
    preds = []
    with torch.no_grad():
        for i in range(0, slm.shape[0], chunk):
            preds.append(tm.forward_model_slm(slm[i:i + chunk]))
    pred = torch.cat(preds)                                               # (n, 2, H, W)
    n, _, h, w = pred.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij")
    support = ((yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2 <= SUPPORT_RADIUS ** 2).reshape(-1)
    out = {"n_held_out": int(n)}
    for ip, plane in enumerate(("image", "fourier")):
        p, t = pred[:, ip].reshape(n, -1), target[:, ip].reshape(n, -1)
        full = _pcc_rows(p, t)
        inside = _pcc_rows(p[:, support], t[:, support])
        pooled = _pcc_rows(p.reshape(1, -1), t.reshape(1, -1))[0]
        pooled_amp = _pcc_rows(p.sqrt().reshape(1, -1), t.sqrt().reshape(1, -1))[0]
        out[plane] = {"pcc_full_frame": float(full.mean()), "pcc_full_frame_std_over_masks": float(full.std()),
                      "pcc_in_support": float(inside.mean()), "pcc_pooled_all_frames": float(pooled),
                      "pcc_pooled_amplitude": float(pooled_amp)}
    return out


def check_b0(pipeline, test_data) -> dict:
    """b = 0 in the copied code reproduces the forward model of the current checkpoint."""
    orig = torch.load(CURRENT_CHECKPOINT, map_location=pipeline.cfg.device if torch.cuda.is_available() else "cpu", weights_only=False)
    orig.device = "cuda" if torch.cuda.is_available() else "cpu"
    orig.eval()
    copy = build_system(pipeline, 0.0)
    missing = copy.load_state_dict(orig.state_dict(), strict=True)
    copy.eval()
    slm = test_data["slm_patterns"][:16].to(copy.device)
    with torch.no_grad():
        i_o, i_c = orig.forward_model_slm(slm), copy.forward_model_slm(slm)
        e_o = orig.forward_model_slm(slm, return_complex_field=True)
        e_c = copy.forward_model_slm(slm, return_complex_field=True)
        copy.fourier_pol_weights = (1.4, 0.6)
        i_w = copy.forward_model_slm(slm)
        manual = 1.4 * e_c[:, 1, 0].abs() ** 2 + 0.6 * e_c[:, 1, 1].abs() ** 2
    res = {"class_of_checkpoint": type(orig).__module__ + "." + type(orig).__name__,
           "calibrated_fast_path": bool(getattr(orig, "_calibrated", False)),
           "state_dict_load": str(missing),
           "max_abs_diff_intensity_b0": float((i_o - i_c).abs().max()), "max_intensity": float(i_o.abs().max()),
           "max_abs_diff_field_b0": float((e_o - e_c).abs().max()),
           "image_plane_unchanged_by_weights": float((i_w[:, 0] - i_o[:, 0]).abs().max()),
           "weighted_fourier_rel_diff_to_manual": float(((i_w[:, 1] - manual).abs().max() / manual.abs().max())),
           "held_out_current_checkpoint": held_out(orig, test_data)}
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--check", action="store_true", help="only check that b = 0 reproduces the current model")
    ap.add_argument("--epochs", type=int, default=0, help="0 = the original number (100)")
    ap.add_argument("--seed", type=int, default=42, help="torch seed (batch order); the original run was not seeded")
    ap.add_argument("--save-dir", default="", help="folder for the full pickled TM (about 490 MB); empty = do not save")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = rtf.PipelineConfig()
    pipeline = rtf.TMFocusPipeline(cfg)                                   # loads geometry files only; setup() is never called
    train_data, test_data = load_data(cfg, pipeline)

    if args.check:
        json.dump(check_b0(pipeline, test_data), open(OUT_DIR / "camweights_check_b0.json", "w"), indent=1)
        return

    tm = build_system(pipeline, args.b)
    for d in (train_data, test_data):
        d["slm_patterns"] = d["slm_patterns"].to(tm.device)
        d["intensity_targets"] = d["intensity_targets"].to(tm.device)
    epochs = int(args.epochs) if args.epochs > 0 else int(cfg.tm_num_epochs)
    torch.manual_seed(args.seed)
    t0 = time.time()
    history = rtf.optimize_tm_slm(
        tm_system=tm, slm_patterns=train_data["slm_patterns"], intensity_targets=train_data["intensity_targets"],
        num_epochs=epochs, learning_rate=cfg.tm_learning_rate, batch_size=cfg.tm_batch_size, planes=cfg.optimize_planes,
        optimize_aberrations=cfg.optimize_aberrations, test_data=test_data,
        use_exposure_normalization=cfg.use_exposure_normalization)
    minutes = (time.time() - t0) / 60.0
    tm.eval()
    res = {"b": float(args.b), "fourier_pol_weights": list(tm.fourier_pol_weights), "epochs": epochs, "seed": args.seed,
           "minutes": minutes, "held_out": held_out(tm, test_data),
           "final_train_loss": float(history["epoch_losses"][-1]), "final_test_loss": float(history["test_losses"][-1]),
           "train_losses": [float(v) for v in history["epoch_losses"]], "test_losses": [float(v) for v in history["test_losses"]]}
    tag = args.tag or f"b{args.b:+.2f}"
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for p in tm.parameters():
            p.requires_grad_(False)
        path = save_dir / f"tm_system_checkpoint_camweights_{tag}.pt"
        torch.save(tm, path)
        res["checkpoint"] = str(path)
    json.dump(res, open(OUT_DIR / f"camweights_{tag}.json", "w"), indent=1)
    ho = res["held_out"]
    print(f"RESULT b = {args.b:+.2f}: held-out correlation image {ho['image']['pcc_full_frame']:.4f}, Fourier {ho['fourier']['pcc_full_frame']:.4f} "
          f"(in support: {ho['image']['pcc_in_support']:.4f}, {ho['fourier']['pcc_in_support']:.4f}); "
          f"final train loss {res['final_train_loss']:.4f}, test loss {res['final_test_loss']:.4f}; {minutes:.1f} min")


if __name__ == "__main__":
    main()
