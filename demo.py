"""
Synthetic end-to-end demo of reference-less modal TM retrieval.

A small step-index fiber is modelled in the guided-mode basis. Dual-plane,
polarization-summed intensities (I = |E_x|² + |E_y|²) are synthesized from a
known ground-truth TM with no interferometer, then Algorithm 1 (modal ADMM)
retrieves the TM. Quality is reported as the agreement between predicted and
measured intensities (Pearson correlation), in the plane the fit used and in
the conjugate plane it never saw.

No H5 data and no GPU are required. See README.md and
the manuscript (Algorithm 1 = ADMM retrieval, Algorithm 2 = Adam calibration).

Reproducible numbers: torch/numpy RNG are seeded. Exact floats can still vary
with BLAS and CPU vs GPU.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics_losses import PearsonCorrelationLoss
from TM_retrieve import TMMeasurementSystem
from TM_retrieval_algos import optimize_tm_admm_dual_plane

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "mode_fields"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "demo"


def small_fiber_params() -> Tuple[Dict, Dict]:
    """Tens of modes on a coarse grid — not the paper's 1508-mode / 128×128 fiber."""
    n1 = 1.457
    na = 0.22
    fiber_params = {
        "a": 5e-6,
        "n1": n1,
        "n0": float(np.sqrt(n1**2 - na**2)),
        "wavelength": 632.8e-9,
    }
    grid_params = {
        "x_range": (-8e-6, 8e-6),
        "y_range": (-8e-6, 8e-6),
        "nx": 32,
        "ny": 32,
    }
    return fiber_params, grid_params


def build_synthetic_system(
    device: str = "cpu",
    cache_dir: str | os.PathLike | None = None,
    seed: int = 0,
    zernike_n_modes: int = 4,
) -> TMMeasurementSystem:
    """Build a small TMMeasurementSystem with lab_data=False (known real_tm_tensor)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    fiber_params, grid_params = small_fiber_params()
    n_pix = grid_params["nx"]
    zoomfft_params = {
        side: {
            "a_init": 1.0,
            "n0_init": (n_pix - 1) / 2,
            "m0_init": (n_pix - 1) / 2,
            "k0_init": (n_pix - 1) / 2,
            "l0_init": (n_pix - 1) / 2,
        }
        for side in ("slm", "tran")
    }
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    system = TMMeasurementSystem(
        fiber_params,
        grid_params,
        device=device,
        zoomfft_params=zoomfft_params,
        zernike_n_modes=zernike_n_modes,
        lab_data=False,
        cache_dir=str(cache_dir),
    )
    return system


def synthesize_measurements(
    tm_system: TMMeasurementSystem,
    num_measurements: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random SLM phases and dual-plane intensities from the ground-truth TM."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return tm_system.sample_measurements_slm(
        num_measurements=num_measurements,
        use_aberrations=False,
    )


def retrieve_admm(
    tm_system: TMMeasurementSystem,
    slm_patterns: torch.Tensor,
    intensity_targets: torch.Tensor,
    num_epochs: int,
    planes: str = "both",
) -> Dict:
    # calibrate() has already reset optimized_tm via reinitialize_optimized_tm().
    z_in = tm_system.slm_to_input_modes(slm_patterns)
    return optimize_tm_admm_dual_plane(
        tm_system,
        z_in,
        intensity_targets,
        num_epochs=num_epochs,
        planes=planes,
        init_method="default_beta_prior",
    )


def plane_pcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    plane: int,
) -> float:
    loss_fn = PearsonCorrelationLoss()
    return float(1.0 - loss_fn(pred[:, plane], target[:, plane]).item())


def save_demo_figure(
    tm_system: TMMeasurementSystem,
    history: Dict,
    slm_patterns: torch.Tensor,
    intensity_targets: torch.Tensor,
    output_path: Path,
) -> None:
    with torch.no_grad():
        pred = tm_system.forward_model_slm(slm_patterns[:1])
    meas = intensity_targets[0].detach().cpu().numpy()
    pred_np = pred[0].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11.0))
    axes[0, 0].plot(history["epoch_losses"], color="C0")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("ADMM intensity MSE")
    axes[0, 0].set_title("Loss")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("probe 0")

    for col, (title, img) in enumerate(
        (("measured image", meas[0]), ("predicted image", pred_np[0]))
    ):
        axes[1, col].imshow(img, cmap="gray")
        axes[1, col].set_title(title)
        axes[1, col].axis("off")
    for col, (title, img) in enumerate(
        (("measured Fourier", meas[1]), ("predicted Fourier", pred_np[1]))
    ):
        axes[2, col].imshow(img, cmap="gray")
        axes[2, col].set_title(title)
        axes[2, col].axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.04, hspace=0.35, wspace=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_demo(
    device: str = "cpu",
    epochs: int = 20,
    num_measurements: int | None = None,
    seed: int = 0,
    output_dir: str | os.PathLike = DEFAULT_OUTPUT_DIR,
    cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
    budget_multiple: float = 2.0,
) -> Dict:
    print("1. Build a small synthetic system (lab_data=False).")
    tm_system = build_synthetic_system(device=device, cache_dir=cache_dir, seed=seed)
    n_modes = len(tm_system.guided_modes)
    print(f"   Guided modes found: {n_modes}")
    print(f"   Mode-field cache directory: {cache_dir}")

    print("2. Calibrate the linear output path.")
    print("   Note: calibrate() resets the TM via reinitialize_optimized_tm().")
    tm_system.calibrate()
    report = tm_system.validate_aberrated_mode_basis()
    print(
        "   validate_aberrated_mode_basis: "
        f"ok={report.get('ok')}, calibrated={report['calibrated']}, "
        f"output_buffers={report['output_calibrated_buffers_present']}, "
        f"matches_recompute={report['output_calibrated_matches_recompute']} "
        "(ok=False is expected here: Zernike coefficients are still zero)"
    )

    if num_measurements is None:
        num_measurements = int(round(budget_multiple * n_modes))
    print(
        f"3. Synthesize measurements: {num_measurements} probes "
        f"({num_measurements / n_modes:.2f} × M, M={n_modes})."
    )
    slm_patterns, intensity_targets = synthesize_measurements(
        tm_system, num_measurements=num_measurements, seed=seed + 1
    )
    print(f"   intensity shape {tuple(intensity_targets.shape)} (batch, plane, H, W); 0=image, 1=Fourier")

    print("4. Retrieve with optimize_tm_admm_dual_plane (init_method='default_beta_prior').")
    history = retrieve_admm(
        tm_system, slm_patterns, intensity_targets, num_epochs=epochs, planes="both"
    )
    with torch.no_grad():
        pred_both = tm_system.forward_model_slm(slm_patterns)
    pcc_both_image = plane_pcc(pred_both, intensity_targets, 0)
    pcc_both_fourier = plane_pcc(pred_both, intensity_targets, 1)

    print("5. Report (agreement between predicted and measured intensities).")
    print(f"   Loss epoch 0: {history['epoch_losses'][0]:.6e}")
    print(f"   Loss last:    {history['epoch_losses'][-1]:.6e}")
    print(f"   PCC image plane:   {pcc_both_image:.4f}")
    print(f"   PCC Fourier plane: {pcc_both_fourier:.4f}")

    output_dir = Path(output_dir)
    fig_path = output_dir / "demo_summary.png"
    save_demo_figure(tm_system, history, slm_patterns, intensity_targets, fig_path)
    print(f"   Figure written to {fig_path}")

    print("6. Cross-plane miniature: fit on image only, score the unseen Fourier plane.")
    tm_system.reinitialize_optimized_tm()
    history_img = retrieve_admm(
        tm_system, slm_patterns, intensity_targets, num_epochs=epochs, planes="image"
    )
    with torch.no_grad():
        pred = tm_system.forward_model_slm(slm_patterns)
    pcc_image = plane_pcc(pred, intensity_targets, 0)
    pcc_fourier = plane_pcc(pred, intensity_targets, 1)
    print(f"   PCC on image plane (seen):    {pcc_image:.4f}")
    print(f"   PCC on Fourier plane (unseen): {pcc_fourier:.4f}")
    print(f"7. Done. Figure: {fig_path}")

    return {
        "n_modes": n_modes,
        "num_measurements": num_measurements,
        "pcc_both_image": pcc_both_image,
        "pcc_both_fourier": pcc_both_fourier,
        "loss_first": float(history["epoch_losses"][0]),
        "loss_last": float(history["epoch_losses"][-1]),
        "pcc_image": pcc_image,
        "pcc_fourier": pcc_fourier,
        "history": history,
        "history_image_only": history_img,
        "figure_path": str(fig_path),
        "tm_system": tm_system,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic dual-plane modal ADMM demo (CPU, no data).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-measurements", type=int, default=0, help="0 = budget_multiple × M")
    parser.add_argument("--budget-multiple", type=float, default=2.0, help="Measurement budget as a multiple of M")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smallest settings that still show the TM error decreasing.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    epochs = args.epochs
    budget = args.budget_multiple
    n_meas = args.num_measurements if args.num_measurements > 0 else None
    if args.quick:
        epochs = min(epochs, 12)
        budget = 1.5
        if args.num_measurements <= 0:
            n_meas = None
    run_demo(
        device=args.device,
        epochs=epochs,
        num_measurements=n_meas,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        budget_multiple=budget,
    )


if __name__ == "__main__":
    main()
