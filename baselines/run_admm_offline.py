"""
Offline ADMM benchmark: retrieve the scalar pixel-basis TM from saved lab measurements using
the ADMM phase-retrieval algorithm of Wei et al. (Phys. Rev. Applied 25, 064049 (2026), see
`admm.pdf`), loop over probe-count subsets, optimize SLM focus masks after each ADMM epoch, and
save the masks that maximize the focus metric for later lab projection (no hardware required).

This mirrors `run_raf_offline.py` (same data, same subsets, same focus-mask optimization and
output schema); only the TM-retrieval algorithm differs. We control a single input polarization,
so the paper's coupled vectorial model c = |A x + B y| reduces to the scalar model c = |A x|.

Uses image-plane intensities only (transmission_image); Fourier-plane data is ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from admm_tm import ADMMOptions, admm_tm_epochs
from motor_pos_utils import (
    DEFAULT_NUM_RANDOM_TRIALS,
    mask_run_label,
    sample_random_subset_indices,
)
from metrics_losses import PBR_DEFINITION
from run_raf_offline import (
    RafSlmMaskWriter,
    _subset_sizes_for_n_slm_pixels,
    build_raf_tensors,
    load_raf_h5_data,
    load_slm_geometry,
    optimize_raf_focus_mask,
    phase_to_uint8,
    save_results_tables,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class OfflineAdmmConfig:
    slm_n_out: int = 32
    n_out: int = 128
    slm_active_extent_px: int = 640
    random_seed: int = 42
    device: str = "cuda"
    admm_iters: int = 30
    admm_mu: float = 5.0
    admm_rho: float = 4.0
    admm_npower_iter: int = 30
    admm_init_keep_ratio: float = 5.0 / 6.0
    admm_ridge: float = 1e-6
    slm_opt_iters: int = 150
    slm_opt_lr: float = 0.1
    focus_center_radius_px: int = 1
    n_samples: int = 0
    h5_group: str = "experiment_0"
    num_random_trials: int = DEFAULT_NUM_RANDOM_TRIALS
    admm_subset_sizes: Optional[List[int]] = None
    output_dir: str = "results/offline/admm"
    output_h5_path: str = "data/measurement_raf.h5"
    slm_masks_h5_path: str = "results/offline/admm/slm_focus_masks.h5"
    pm_center_x_path: str = "PM_center_x.npy"
    pm_center_y_path: str = "PM_center_y.npy"

    @property
    def slm_ratio(self) -> int:
        return self.slm_active_extent_px // self.slm_n_out


def run_offline(cfg: OfflineAdmmConfig) -> Dict:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = cfg.device
    if runtime_device == "cuda" and not torch.cuda.is_available():
        runtime_device = "cpu"

    n_slm_pixels = cfg.slm_n_out * cfg.slm_n_out
    subset_sizes = cfg.admm_subset_sizes or _subset_sizes_for_n_slm_pixels(n_slm_pixels)
    max_subset = max(subset_sizes)

    with h5py.File(cfg.output_h5_path, "r") as f:
        if cfg.h5_group not in f:
            raise ValueError(f"Group '{cfg.h5_group}' not found in {cfg.output_h5_path}")
        n_available = int(f[cfg.h5_group]["PhaseMask"].shape[0])
    n_samples = cfg.n_samples if cfg.n_samples > 0 else n_available
    if n_samples < max_subset:
        raise ValueError(f"Need at least {max_subset} samples, got {n_samples} (available {n_available})")

    summary: Dict = {
        "focus_metric_definition": PBR_DEFINITION,
        "subset_runs": {},
        "num_random_trials": cfg.num_random_trials,
        "offline": True,
        "algorithm": "admm",
        "n_slm_pixels": n_slm_pixels,
        "slm_n_out": cfg.slm_n_out,
        "n_out": cfg.n_out,
        "n_samples": n_samples,
        "subset_sizes": subset_sizes,
    }

    geometry = load_slm_geometry(cfg)
    if geometry is None:
        print(
            "SLM geometry files not found (PM_center_x/y.npy). "
            f"Saving {cfg.slm_n_out}x{cfg.slm_n_out} phase masks only; pack to full SLM in the lab."
        )
    else:
        print("SLM geometry loaded; saving both small and full (1200x1920) phase masks.")

    full_data = load_raf_h5_data(cfg.output_h5_path, cfg.h5_group, n_samples=n_samples, im_factor=1.0)
    full_data["slm_phase"] = full_data["slm_phase"].to(runtime_device)
    full_data["intensity_image"] = full_data["intensity_image"].to(runtime_device)

    mask_writer = RafSlmMaskWriter(
        h5_path=Path(cfg.slm_masks_h5_path),
        geometry=geometry,
        slm_n_out=cfg.slm_n_out,
    )

    opts = ADMMOptions(
        mu=cfg.admm_mu,
        rho=cfg.admm_rho,
        iters=cfg.admm_iters,
        npower_iter=cfg.admm_npower_iter,
        init_keep_ratio=cfg.admm_init_keep_ratio,
        ridge=cfg.admm_ridge,
    )
    try:
        for subset_size in subset_sizes:
            subset_summary: Dict[str, Any] = {"trials": {}}
            for trial_idx in range(cfg.num_random_trials):
                run_label = mask_run_label("admm", subset_size, trial_idx)
                sample_indices = sample_random_subset_indices(
                    n_samples, subset_size, trial_idx, cfg.random_seed
                )
                index_tensor = torch.as_tensor(sample_indices, dtype=torch.long, device=runtime_device)
                print(
                    f"--- ADMM run: subset_size={subset_size}, "
                    f"trial={trial_idx + 1}/{cfg.num_random_trials} ---"
                )
                gen = torch.Generator(device=runtime_device).manual_seed(cfg.random_seed + trial_idx)
                slm_slice = full_data["slm_phase"][index_tensor]
                intensity_slice = full_data["intensity_image"][index_tensor]
                y_intensity, x_probes = build_raf_tensors(slm_slice, intensity_slice)

                simulated_focus_metrics: List[float] = []
                epoch_times_acc: List[float] = []
                ls_losses_acc: List[float] = []

                def on_epoch_end(epoch: int, epoch_seconds: float, a_pixel: torch.Tensor, ls_loss: float) -> None:
                    optimized_phase, focus_metric = optimize_raf_focus_mask(
                        a_pixel=a_pixel,
                        n_pix_slm=cfg.slm_n_out,
                        n_pix_out=cfg.n_out,
                        num_iterations=cfg.slm_opt_iters,
                        lr=cfg.slm_opt_lr,
                        center_radius=cfg.focus_center_radius_px,
                    )
                    phase_small_uint8 = phase_to_uint8(optimized_phase.detach().cpu().numpy())
                    mask_writer.append(run_label, epoch, phase_small_uint8, focus_metric)
                    simulated_focus_metrics.append(float(focus_metric))
                    epoch_times_acc.append(float(epoch_seconds))
                    ls_losses_acc.append(float(ls_loss))
                    print(
                        f"[{run_label}] epoch {epoch + 1}/{cfg.admm_iters} | "
                        f"time={epoch_seconds:.3f}s | ls_loss={ls_loss:.4f} | simulated_PBR={focus_metric:.2f}"
                    )

                _, history = admm_tm_epochs(
                    y_intensity,
                    x_probes,
                    opts,
                    generator=gen,
                    on_epoch_end=on_epoch_end,
                )
                corr_acc = [float(c) for c in history.get("corr", [])]
                setup_time_sec = float(history["setup_time_sec"][0])
                initialization_time_sec = float(history["initialization_time_sec"][0])
                subset_summary["trials"][str(trial_idx)] = {
                    "run_label": run_label,
                    "sample_indices": sample_indices.tolist(),
                    "ls_loss": ls_losses_acc,
                    "epoch_times_sec": epoch_times_acc,
                    "admm_setup_time_sec": setup_time_sec,
                    "admm_initialization_time_sec": initialization_time_sec,
                    "simulated_focus_metrics": simulated_focus_metrics,
                    "admm_corr": corr_acc,
                    "best_simulated_focus_epoch": (
                        simulated_focus_metrics.index(max(simulated_focus_metrics)) + 1
                        if simulated_focus_metrics
                        else None
                    ),
                }
            summary["subset_runs"][str(subset_size)] = subset_summary
    finally:
        mask_writer.close()

    summary["slm_masks_h5_path"] = str(cfg.slm_masks_h5_path)
    summary["config"] = asdict(cfg)
    summary_path = output_dir / "admm_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    table_paths = save_results_tables(summary, output_dir, name_prefix="admm_benchmark", title="ADMM")
    summary["table_paths"] = {k: str(v) for k, v in table_paths.items()}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Summary: {summary_path}")
    print(f"SLM masks: {cfg.slm_masks_h5_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ADMM TM-retrieval benchmark from saved H5 measurements")
    parser.add_argument("--h5-path", type=str, default="data/measurement_raf.h5")
    parser.add_argument("--h5-group", type=str, default="experiment_0")
    parser.add_argument("--output-dir", type=str, default="results/offline/admm")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--admm-iters", type=int, default=250)
    parser.add_argument("--admm-mu", type=float, default=5.0)
    parser.add_argument("--admm-rho", type=float, default=4.0)
    parser.add_argument("--admm-npower-iter", type=int, default=30)
    parser.add_argument("--admm-init-keep-ratio", type=float, default=5.0 / 6.0)
    parser.add_argument("--admm-ridge", type=float, default=1e-6)
    parser.add_argument("--slm-opt-iters", type=int, default=250)
    parser.add_argument("--slm-opt-lr", type=float, default=0.1)
    parser.add_argument("--focus-center-radius", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=0, help="0 = use all samples in the H5 group")
    parser.add_argument(
        "--admm-subset-sizes",
        type=str,
        default="",
        help="Comma-separated subset sizes (default: 1.2N, 1.5N, 2N, 3N with N=slm_n_out^2)",
    )
    parser.add_argument("--slm-n-out", type=int, default=32)
    parser.add_argument("--n-out", type=int, default=128)
    parser.add_argument("--pm-center-x", type=str, default="PM_center_x.npy")
    parser.add_argument("--pm-center-y", type=str, default="PM_center_y.npy")
    parser.add_argument(
        "--num-random-trials",
        type=int,
        default=DEFAULT_NUM_RANDOM_TRIALS,
        help="Number of random subset index draws per subset size (default: 10)",
    )
    args = parser.parse_args()

    cfg = OfflineAdmmConfig()
    cfg.output_h5_path = args.h5_path
    cfg.h5_group = args.h5_group
    cfg.output_dir = args.output_dir
    cfg.slm_masks_h5_path = str(Path(args.output_dir) / "slm_focus_masks.h5")
    cfg.device = args.device
    cfg.admm_iters = args.admm_iters
    cfg.admm_mu = args.admm_mu
    cfg.admm_rho = args.admm_rho
    cfg.admm_npower_iter = args.admm_npower_iter
    cfg.admm_init_keep_ratio = args.admm_init_keep_ratio
    cfg.admm_ridge = args.admm_ridge
    cfg.slm_opt_iters = args.slm_opt_iters
    cfg.slm_opt_lr = args.slm_opt_lr
    cfg.focus_center_radius_px = args.focus_center_radius
    cfg.n_samples = args.n_samples
    cfg.slm_n_out = args.slm_n_out
    cfg.n_out = args.n_out
    cfg.pm_center_x_path = args.pm_center_x
    cfg.pm_center_y_path = args.pm_center_y
    cfg.num_random_trials = args.num_random_trials
    if args.admm_subset_sizes.strip():
        cfg.admm_subset_sizes = [int(x.strip()) for x in args.admm_subset_sizes.split(",") if x.strip()]

    run_offline(cfg)


if __name__ == "__main__":
    main()
