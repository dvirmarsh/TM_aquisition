"""
Stage B (revised, 15 Nov 2026), step 2 (offline, no hardware): TM retrieval with the Fourier-camera
weights (1 + b, 1 - b), b = tanh(beta), LEARNED by gradient descent together with the TM.

Forward model (TM_retrieve_camweights.py, a copy of TM_retrieve.py):
    image camera:    |E_x|^2 + |E_y|^2                     (fixed weights (1, 1))
    Fourier camera:  (1 + b) |E_x|^2 + (1 - b) |E_y|^2
Everything else is the original run: data loading and train/test split (split 0.95, seed 42),
initialization (TMFocusPipeline.initialize_tm_system), optimizer and loss
(run_tm_focus_optimization.optimize_tm_slm: Adam, lr 0.01, batch 512, both planes). beta joins the
parameter group of the TM (lr 0.01). run_tm_focus_optimization.py and TM_retrieve.py are not modified.

    python retrieve_tm_learn_b.py --b-init 0.47 --epochs 200 --save-dir <folder>      # learn b
    python retrieve_tm_learn_b.py --fixed-b 0.47 --epochs 200 --save-dir <folder>     # b held fixed
    python retrieve_tm_learn_b.py --scale-check                                       # loss does not depend on the plane scale

Scale between the planes: the loss is 1 - Pearson correlation, computed per plane and averaged.
It does not change when one plane is multiplied by a constant, so a plane scale cannot be learned
by this loss and cannot bias b. Instead the script reports, after the retrieval, the least-squares
plane scale: the factor by which the present constant (im_factor in TMFocusPipeline.run) would
have to be multiplied for one common gain to fit both planes (1 = the present constant is right).

Output: tm_focus_outputs/stageB_offline/learn_b_<tag>.json (+ the pickled TM in --save-dir).
"Held-out correlation" as in retrieve_tm_camera_weights.py (302 held-out masks, full frame).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import retrieve_tm_camera_weights as rcw           # noqa: E402  (stubs create_refs: no hardware library is loaded)

rtf = rcw.rtf
OUT_DIR = TM_LIB / "tm_focus_outputs" / "stageB_offline"


def fitted_plane_scale(tm, test_data, chunk: int = 32) -> dict:
    """Least-squares gain of the model to the held-out frames, per plane, and their ratio."""
    slm = test_data["slm_patterns"].to(tm.device)
    target = test_data["intensity_targets"].to(tm.device)
    with torch.no_grad():
        pred = torch.cat([tm.forward_model_slm(slm[i:i + chunk]) for i in range(0, slm.shape[0], chunk)])
    gain = [(float((pred[:, p] * target[:, p]).sum() / (pred[:, p] ** 2).sum())) for p in (0, 1)]
    return {"gain_image": gain[0], "gain_fourier": gain[1], "plane_scale_relative_to_present_constant": gain[1] / gain[0]}


def scale_check(pipeline, train_data) -> None:
    tm = rcw.build_system(pipeline, 0.3)
    crit = rtf.PearsonCorrelationLoss(dim=(0, 2, 3), center=True)
    slm = train_data["slm_patterns"][:64].to(tm.device)
    tgt = train_data["intensity_targets"][:64].to(tm.device)
    with torch.no_grad():
        pred = tm.forward_model_slm(slm)
        base = float(crit(pred, tgt))
        scaled_t = tgt.clone(); scaled_t[:, 0] *= 3.0
        scaled_p = pred.clone(); scaled_p[:, 1] *= 0.2
        print(f"loss {base:.8f} | image targets x 3: {float(crit(pred, scaled_t)):.8f} | Fourier prediction x 0.2: {float(crit(scaled_p, tgt)):.8f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-init", type=float, default=None, help="learn b, starting from this value")
    ap.add_argument("--fixed-b", type=float, default=None, help="hold b fixed at this value")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42, help="torch seed (batch order)")
    ap.add_argument("--save-dir", default="", help="folder for the pickled TM (about 490 MB); empty = do not save")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scale-check", action="store_true")
    ap.add_argument("--h5", default="", help="dataset file (default: measurement_focus.h5 of the pipeline)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = rtf.PipelineConfig()
    pipeline = rtf.TMFocusPipeline(cfg)                                   # geometry files only; setup() is never called
    if args.h5:
        cfg.output_h5_path = str(Path(args.h5).resolve())
    train_data, test_data = rcw.load_data(cfg, pipeline)
    if args.scale_check:
        scale_check(pipeline, train_data)
        return
    if (args.b_init is None) == (args.fixed_b is None):
        raise SystemExit("Give exactly one of --b-init (learn b) and --fixed-b.")

    learn = args.b_init is not None
    b0 = float(args.b_init if learn else args.fixed_b)
    tm = rcw.build_system(pipeline, b0)                                   # fixed weights (1 + b0, 1 - b0)
    if learn:
        tm.enable_learnable_fourier_weights(b0)                           # from here on b = tanh(beta) is a parameter
    for d in (train_data, test_data):
        d["slm_patterns"] = d["slm_patterns"].to(tm.device)
        d["intensity_targets"] = d["intensity_targets"].to(tm.device)

    # b after every epoch: the only forward call without gradients in optimize_tm_slm is the held-out evaluation
    b_trace = []
    forward = tm.forward_model_slm

    def traced_forward(*a, **k):
        if not torch.is_grad_enabled():
            b_trace.append(tm.fourier_weight_b())
        return forward(*a, **k)

    tm.forward_model_slm = traced_forward
    torch.manual_seed(args.seed)
    t0 = time.time()
    history = rtf.optimize_tm_slm(
        tm_system=tm, slm_patterns=train_data["slm_patterns"], intensity_targets=train_data["intensity_targets"],
        num_epochs=int(args.epochs), learning_rate=cfg.tm_learning_rate, batch_size=cfg.tm_batch_size, planes=cfg.optimize_planes,
        optimize_aberrations=cfg.optimize_aberrations, test_data=test_data,
        use_exposure_normalization=cfg.use_exposure_normalization)
    minutes = (time.time() - t0) / 60.0
    del tm.forward_model_slm                                              # back to the class method (needed for pickling)
    tm.eval()
    for p in tm.parameters():
        p.requires_grad_(False)
    b_final = tm.fourier_weight_b()
    tm.fourier_pol_weights = (1.0 + b_final, 1.0 - b_final)               # kept in step with the learned b
    res = {"mode": "learned" if learn else "fixed", "b_init": b0, "b": b_final, "epochs": int(args.epochs), "seed": args.seed,
           "minutes": minutes, "held_out": rcw.held_out(tm, test_data), "plane_scale": fitted_plane_scale(tm, test_data),
           "final_train_loss": float(history["epoch_losses"][-1]), "final_test_loss": float(history["test_losses"][-1]),
           "b_per_epoch": b_trace, "train_losses": [float(v) for v in history["epoch_losses"]],
           "test_losses": [float(v) for v in history["test_losses"]]}
    tag = args.tag or (f"learn_init{b0:+.2f}" if learn else f"fixed{b0:+.2f}")
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"tm_system_checkpoint_camweights_{tag}.pt"
        torch.save(tm, path)
        res["checkpoint"] = str(path)
    json.dump(res, open(OUT_DIR / f"learn_b_{tag}.json", "w"), indent=1)
    ho = res["held_out"]
    print(f"RESULT {tag}: b {b0:+.3f} -> {b_final:+.4f} ({res['mode']}); held-out correlation image {ho['image']['pcc_full_frame']:.4f}, "
          f"Fourier {ho['fourier']['pcc_full_frame']:.4f}; fitted plane scale / present constant {res['plane_scale']['plane_scale_relative_to_present_constant']:.4f}; "
          f"train loss {res['final_train_loss']:.4f}, test loss {res['final_test_loss']:.4f}; {minutes:.1f} min")


if __name__ == "__main__":
    main()
