"""
Offline modal TM retrieval: Adam joint calibration, calibrate(), then ADMM
subset benchmarks on post-motor data, saving SLM focus masks for later lab
projection (no hardware required).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

TM_LIB_DIR = Path(__file__).resolve().parent
if str(TM_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(TM_LIB_DIR))

from TM_retrieve import TMMeasurementSystem
from TM_retrieval_algos import (
    optimize_tm_admm_dual_plane,
    optimize_tm_slm,
)
from motor_pos_utils import (
    CALIBRATION_H5_GROUP,
    DEFAULT_NUM_RANDOM_TRIALS,
    mask_run_label,
    parse_mask_run_label,
    sample_random_subset_indices,
)
from utils import load_real_mmf_data
from metrics_losses import PBR_DEFINITION, nominal_fiber_support_mask, peak_to_background_ratio


def _subset_sizes_for_n_modes(n_modes: int) -> List[int]:
    return [int(1.2 * n_modes), int(1.5 * n_modes), 2 * n_modes, 3 * n_modes]


def _pcc_uses_single_plane(planes: str) -> bool:
    return planes.lower() in ("image", "fourier")


def _focus_eval_plane_for_pcc_planes(planes: str) -> str:
    """Focus SLM masks on the complementary plane when TM uses one plane.

    - TM on Fourier → maximize focus on the image plane (held out).
    - TM on image → maximize focus on the Fourier plane (held out).
    - TM on both → maximize focus on the image plane (default).
    """
    planes_l = planes.lower()
    if planes_l == "image":
        return "fourier"
    return "image"


REPO_ROOT = Path(__file__).resolve().parent
MODE_FIELDS_CACHE_DIR = str(REPO_ROOT / "data" / "mode_fields")


def _safe_index(values: List[Any], idx: int, default: Any = None) -> Any:
    if not values or idx < 0 or idx >= len(values):
        return default
    return values[idx]



def _iter_trial_subset_runs(summary: Dict) -> List[Tuple[int, int, Dict]]:
    """Yield (subset_size, trial_idx, run_dict)."""
    out: List[Tuple[int, int, Dict]] = []
    subset_runs = summary.get("subset_runs") or {}
    for subset_key in sorted(subset_runs.keys(), key=lambda k: int(k)):
        subset_size = int(subset_key)
        run = subset_runs[subset_key]
        trials = run.get("trials") or {}
        if trials:
            for trial_key in sorted(trials.keys(), key=lambda k: int(k)):
                out.append((subset_size, int(trial_key), trials[trial_key]))
        else:
            out.append((subset_size, 0, run))
    return out


def build_epoch_results_rows(summary: Dict) -> List[Dict[str, Any]]:
    n_modes = summary.get("n_guided_modes") or 1
    motor_loc = summary.get("motor_location_mm")
    rows: List[Dict[str, Any]] = []
    for subset_size, trial_idx, run in _iter_trial_subset_runs(summary):
        epoch_times = run.get("epoch_times_sec") or []
        epoch_losses = run.get("epoch_losses") or []
        focus_metrics = run.get("simulated_focus_metrics") or run.get("measured_focus_metrics") or []
        n_epochs = max(len(epoch_times), len(epoch_losses), len(focus_metrics))
        cumulative = 0.0
        for epoch_idx in range(n_epochs):
            epoch_time = float(_safe_index(epoch_times, epoch_idx, 0.0) or 0.0)
            cumulative += epoch_time
            rows.append(
                {
                    "n_guided_modes": n_modes,
                    "motor_location_mm": motor_loc,
                    "subset_size": subset_size,
                    "trial": trial_idx,
                    "subset_size_over_N": round(subset_size / n_modes, 4),
                    "epoch": epoch_idx + 1,
                    "epoch_time_sec": round(epoch_time, 4),
                    "cumulative_epoch_time_sec": round(cumulative, 4),
                    "pcc_loss_1_minus_pcc": round(float(_safe_index(epoch_losses, epoch_idx, float("nan"))), 6),
                    "simulated_center_focus_metric": round(
                        float(_safe_index(focus_metrics, epoch_idx, float("nan"))), 6
                    ),
                }
            )
    return rows


def build_subset_summary_rows(summary: Dict) -> List[Dict[str, Any]]:
    n_modes = summary.get("n_guided_modes") or 1
    motor_loc = summary.get("motor_location_mm")
    rows: List[Dict[str, Any]] = []
    for subset_size, trial_idx, run in _iter_trial_subset_runs(summary):
        epoch_times = [float(v) for v in (run.get("epoch_times_sec") or [])]
        epoch_losses = [float(v) for v in (run.get("epoch_losses") or [])]
        focus_metrics = [
            float(v)
            for v in (run.get("simulated_focus_metrics") or run.get("measured_focus_metrics") or [])
        ]
        rows.append(
            {
                "n_guided_modes": n_modes,
                "motor_location_mm": motor_loc,
                "subset_size": subset_size,
                "trial": trial_idx,
                "subset_size_over_N": round(subset_size / n_modes, 4),
                "n_epochs": max(len(epoch_times), len(epoch_losses), len(focus_metrics)),
                "total_epoch_time_sec": round(sum(epoch_times), 4) if epoch_times else None,
                "mean_epoch_time_sec": round(float(np.mean(epoch_times)), 4) if epoch_times else None,
                "initial_pcc_loss": round(epoch_losses[0], 6) if epoch_losses else None,
                "final_pcc_loss": round(epoch_losses[-1], 6) if epoch_losses else None,
                "best_pcc_loss": round(min(epoch_losses), 6) if epoch_losses else None,
                "initial_simulated_focus": round(focus_metrics[0], 6) if focus_metrics else None,
                "final_simulated_focus": round(focus_metrics[-1], 6) if focus_metrics else None,
                "best_simulated_focus": round(max(focus_metrics), 6) if focus_metrics else None,
            }
        )
    return rows


def _rows_to_markdown_table(rows: List[Dict[str, Any]], max_rows: Optional[int] = None) -> str:
    if not rows:
        return "_No data._"
    display_rows = rows if max_rows is None else rows[:max_rows]
    headers = list(display_rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display_rows:
        cells = ["" if row.get(key) is None else str(row.get(key)) for key in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results_tables(summary: Dict, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = build_epoch_results_rows(summary)
    summary_rows = build_subset_summary_rows(summary)
    paths = {
        "epoch_csv": output_dir / "motor_pcc_benchmark_epoch_table.csv",
        "summary_csv": output_dir / "motor_pcc_benchmark_subset_summary.csv",
        "combined_md": output_dir / "motor_pcc_benchmark_tables.md",
    }
    for path, rows in ((paths["epoch_csv"], epoch_rows), (paths["summary_csv"], summary_rows)):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    paths["combined_md"].write_text(
        "# Motor PCC offline benchmark results\n\n"
        + _rows_to_markdown_table(summary_rows)
        + "\n\n## Per-epoch (first 25)\n\n"
        + _rows_to_markdown_table(epoch_rows, max_rows=25),
        encoding="utf-8",
    )
    print("\n=== PCC subset summary ===")
    print(_rows_to_markdown_table(summary_rows))
    return paths


@dataclass
class OfflineMotorPccConfig:
    n_out: int = 128
    ratio: int = 5
    transmission_image_et: Tuple[float, ...] = (120.0,)
    transmission_fourier_et: Tuple[float, ...] = (150.0,)
    train_test_split_ratio: float = 0.95
    random_seed: int = 42
    device: str = "cuda"
    optimize_aberrations: bool = True
    optimize_planes: str = "both"
    pcc_optimize_planes: str = "both"
    tm_num_epochs: int = 100
    tm_learning_rate: float = 0.01
    tm_batch_size: int = 256
    slm_opt_iters: int = 90
    slm_opt_lr: float = 0.1
    focus_center_radius_px: int = 1
    focus_eval_plane: str = "image"
    n_samples: int = 0
    initial_h5_group: str = CALIBRATION_H5_GROUP
    post_motor_h5_group: str = "post_motor"
    motor_position_mm: float = 5.0
    num_random_trials: int = DEFAULT_NUM_RANDOM_TRIALS
    optimization_method: str = "admm"
    admm_init_method: str = "default_beta_prior"
    admm_init_power_iters: int = 50
    admm_init_num_pixels: int = 500
    admm_init_eig_rel_threshold: float = 1e-3
    pcc_num_epochs: int = 100
    pcc_resync_interval: int = 10
    input_modes_batch_size: int = 64
    use_tm_checkpoint: bool = False
    output_dir: str = "results/offline/modal"
    tm_checkpoint_path: str = "results/offline/modal/tm_system_checkpoint.pt"
    output_h5_path: str = "data/measurement_motor_pcc.h5"
    slm_masks_h5_path: str = "results/offline/modal/slm_focus_masks.h5"
    pm_center_x_path: str = "PM_center_x.npy"
    pm_center_y_path: str = "PM_center_y.npy"
    skip_tm_optimization: bool = False
    skip_pcc: bool = False
    pcc_subset_sizes: Optional[List[int]] = None
    step3_mode: str = "pcc_focus"
    # Step 3 options: "auto" focuses on the plane complementary to a single-plane TM;
    # "image" / "fourier" force the focus plane. With single_plane_subset_sweep, a
    # single-plane TM runs the same subset sweep and random trials as the dual-plane case.
    focus_eval_plane_override: str = "auto"
    single_plane_subset_sweep: bool = False


def initialize_tm_system(cfg: OfflineMotorPccConfig) -> TMMeasurementSystem:
    wavelength = 632.8e-9
    core_index = 1.457
    na = 0.22
    core_radius = 25e-6
    n0 = cfg.n_out
    magnification = 50.0
    obj_f = 200e-3 / magnification

    dx_cam = 3.45e-6
    dx_im = dx_cam / magnification
    dx_four = dx_cam * (250.0 / 200.0)
    dx_slm = 8e-6 * (150.0 / 400.0)
    margin = 1.1
    n_pix_im = round(margin * core_radius / dx_im)
    n_pix_four = round(margin * na * obj_f / dx_four)

    dx_final = {
        "im": dx_im * n_pix_im / 64.0,
        "four": dx_four * n_pix_four / 64.0,
        "slm": dx_slm * 320 / 64.0,
    }
    mult_factor = {
        "tran": dx_final["im"] ** 2 / (wavelength * obj_f),
        "slm": dx_final["slm"] ** 2 / (wavelength * obj_f),
    }
    fiber_params = {
        "a": core_radius,
        "n1": core_index,
        "n0": np.sqrt(core_index**2 - na**2),
        "wavelength": wavelength,
    }
    range_im = (n0 / 2) * dx_final["im"]
    grid_params = {
        "x_range": (-range_im, range_im),
        "y_range": (-range_im, range_im),
        "nx": n0,
        "ny": n0,
    }
    a_param = {
        "slm": 2 * np.pi * dx_final["slm"] * dx_final["im"] / (wavelength * obj_f),
        "tran": 2 * np.pi * dx_final["im"] * dx_final["four"] / (wavelength * obj_f),
    }
    zoomfft_params = {
        side: {
            "a_init": a_param[side],
            "n0_init": (n0 - 1) / 2,
            "m0_init": (n0 - 1) / 2,
            "k0_init": (n0 - 1) / 2,
            "l0_init": (n0 - 1) / 2,
        }
        for side in ["slm", "tran"]
    }
    runtime_device = cfg.device
    if runtime_device == "cuda" and not torch.cuda.is_available():
        runtime_device = "cpu"

    return TMMeasurementSystem(
        fiber_params,
        grid_params,
        device=runtime_device,
        mult_factor=mult_factor,
        zoomfft_params=zoomfft_params,
        zernike_n_modes=8,
        dx=dx_final,
        cache_dir=MODE_FIELDS_CACHE_DIR,
    )


def optimize_slm_pattern_with_tm(
    tm_system: TMMeasurementSystem,
    slm_shape: Tuple[int, int] = (128, 128),
    num_iterations: int = 150,
    lr: float = 0.1,
    center_radius: int = 1,
    target_plane: str = "image",
) -> Tuple[torch.Tensor, float]:
    """
    Maximize the TM-predicted focus at the frame centre (target-window mean over the fiber
    support outside the window) and return the phase with the PBR of its predicted
    intensity (metrics_losses.PBR_DEFINITION).
    """
    for param in tm_system.parameters():
        param.requires_grad = False
    tm_system.eval()

    device = tm_system.device
    slm_phase = nn.Parameter(2 * torch.pi * torch.rand((1, *slm_shape), dtype=torch.float32, device=device))
    optimizer = optim.Adam([slm_phase], lr=lr)
    h = tm_system.grid_params["ny"]
    w = tm_system.grid_params["nx"]
    cy, cx = h // 2, w // 2
    center_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    y_min, y_max = max(0, cy - center_radius), min(h, cy + center_radius + 1)
    x_min, x_max = max(0, cx - center_radius), min(w, cx + center_radius + 1)
    center_mask[y_min:y_max, x_min:x_max] = True
    # Enhancement background: fiber core (image) / NA disk (Fourier) minus the target window.
    support_mask = torch.from_numpy(nominal_fiber_support_mask((h, w))).to(device)
    background_mask = support_mask & ~center_mask
    plane_idx = 0 if target_plane.lower() == "image" else 1

    for _ in range(num_iterations):
        optimizer.zero_grad()
        pred_intensity = tm_system.forward_model_slm(slm_phase)
        plane_intensity = pred_intensity[0, plane_idx]
        center_mean = plane_intensity[center_mask].mean()
        background_mean = plane_intensity[background_mask].mean()
        focus_metric = center_mean / background_mean
        (-focus_metric).backward()
        optimizer.step()

    optimized_phase = torch.remainder(slm_phase.detach(), 2 * torch.pi).squeeze(0)
    with torch.no_grad():
        pred_intensity = tm_system.forward_model_slm(optimized_phase.unsqueeze(0))
        plane_intensity = pred_intensity[0, plane_idx]
    final_pbr, _ = peak_to_background_ratio(
        plane_intensity, support_mask.cpu().numpy(), target_yx=(cy, cx)
    )
    return optimized_phase, final_pbr


def phase_to_uint8(phase_rad: np.ndarray) -> np.ndarray:
    phase_wrapped = np.mod(phase_rad, 2.0 * np.pi)
    return np.floor(phase_wrapped * (256.0 / (2.0 * np.pi))).astype(np.uint8)


def load_slm_geometry(cfg: OfflineMotorPccConfig) -> Optional[Dict]:
    pm_x_path = Path(cfg.pm_center_x_path)
    pm_y_path = Path(cfg.pm_center_y_path)
    if not pm_x_path.exists() or not pm_y_path.exists():
        return None
    return {
        "pm_center_x": int(np.load(pm_x_path).item()),
        "pm_center_y": int(np.load(pm_y_path).item()),
        "ratio": cfg.ratio,
        "n_pix_slm": 320,
    }


def pack_phase_to_full_slm(phase_uint8_small: np.ndarray, geometry: Dict) -> np.ndarray:
    ratio = geometry["ratio"]
    crop_size = geometry["n_pix_slm"]
    pm_center_x = geometry["pm_center_x"]
    pm_center_y = geometry["pm_center_y"]

    full_small = np.zeros((1200 // ratio, 1920 // ratio), dtype=np.uint8)
    y_start_full = pm_center_y - crop_size
    y_end_full = pm_center_y + crop_size
    x_start_full = pm_center_x - crop_size
    x_end_full = pm_center_x + crop_size
    y_start = max(0, y_start_full // ratio)
    y_end = min(full_small.shape[0], (y_end_full + ratio - 1) // ratio)
    x_start = max(0, x_start_full // ratio)
    x_end = min(full_small.shape[1], (x_end_full + ratio - 1) // ratio)
    h = min(y_end - y_start, phase_uint8_small.shape[0])
    w = min(x_end - x_start, phase_uint8_small.shape[1])
    full_small[y_start : y_start + h, x_start : x_start + w] = np.flip(phase_uint8_small[:h, :w])
    return np.kron(full_small, np.ones((ratio, ratio), dtype=np.uint8))


def compute_input_modes(
    tm_system: TMMeasurementSystem,
    slm_patterns: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    slm_patterns = slm_patterns.to(tm_system.device)
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, slm_patterns.shape[0], batch_size):
            end = min(start + batch_size, slm_patterns.shape[0])
            chunk = tm_system.slm_to_input_modes(
                slm_patterns[start:end],
                use_internal_params=True,
                apply_slm_amplitude=True,
                active_polarization="both",
            )
            chunks.append(chunk)
    return torch.cat(chunks, dim=1)


def load_all_mmf_data(h5_path: str, experiment_group: str, im_factor: float) -> Dict[str, torch.Tensor]:
    """Load the full dataset (no train/test split) for PCC subset experiments."""
    print(f"Loading data from {h5_path}, group: {experiment_group}")
    with h5py.File(h5_path, "r") as f:
        if experiment_group not in f:
            raise ValueError(f"Group '{experiment_group}' not found. Available: {list(f.keys())}")
        group = f[experiment_group]
        slm_phase = np.flip(np.array(group["PhaseMask"][:]), axis=(-2, -1))
        intensity_image = np.flip(np.array(group["transmission_image"][:]), axis=(-2, -1))
        intensity_fourier = np.array(group["transmission_fourier"][:])
        if slm_phase.dtype == np.uint8:
            slm_phase = slm_phase.astype(np.float32) * 2 * np.pi / 256.0
        intensity_targets = np.stack([intensity_image, intensity_fourier], axis=1).astype(np.float32)
        intensity_targets[:, 0] *= im_factor
    return {
        "slm_patterns": torch.from_numpy(slm_phase).float(),
        "intensity_targets": torch.from_numpy(intensity_targets).float(),
    }


class SlmMaskWriter:
    def __init__(
        self,
        h5_path: Path,
        geometry: Optional[Dict],
        focus_eval_plane: str,
    ):
        self.h5_path = Path(h5_path)
        self.geometry = geometry
        self.focus_eval_plane = focus_eval_plane
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[h5py.File] = None
        self._reset_labels: set = set()

    def _ensure_group(self, run_label: str) -> h5py.Group:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "a")
        # On the first write to a run_label this session, drop any stale group
        # from a previous run so repeated runs overwrite instead of accumulating.
        if run_label not in self._reset_labels:
            if run_label in self._file:
                del self._file[run_label]
            self._reset_labels.add(run_label)
        if run_label not in self._file:
            grp = self._file.create_group(run_label)
            grp.create_dataset(
                "phase_masks_small_uint8",
                shape=(0, 128, 128),
                maxshape=(None, 128, 128),
                dtype=np.uint8,
                chunks=(1, 128, 128),
            )
            grp.create_dataset(
                "simulated_focus_metrics",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(64,),
            )
            if self.geometry is not None:
                grp.create_dataset(
                    "phase_masks_full_uint8",
                    shape=(0, 1200, 1920),
                    maxshape=(None, 1200, 1920),
                    dtype=np.uint8,
                    chunks=(1, 1200, 1920),
                )
        return self._file[run_label]

    def append(self, run_label: str, epoch: int, phase_small_uint8: np.ndarray, focus_metric: float) -> None:
        grp = self._ensure_group(run_label)
        idx = int(grp["phase_masks_small_uint8"].shape[0])
        grp["phase_masks_small_uint8"].resize(idx + 1, axis=0)
        grp["phase_masks_small_uint8"][idx] = phase_small_uint8
        grp["simulated_focus_metrics"].resize(idx + 1, axis=0)
        grp["simulated_focus_metrics"][idx] = float(focus_metric)
        if self.geometry is not None and "phase_masks_full_uint8" in grp:
            full_mask = pack_phase_to_full_slm(phase_small_uint8, self.geometry)
            grp["phase_masks_full_uint8"].resize(idx + 1, axis=0)
            grp["phase_masks_full_uint8"][idx] = full_mask
        grp.attrs["focus_eval_plane"] = self.focus_eval_plane
        grp.attrs["focus_metric_definition"] = PBR_DEFINITION
        grp.attrs["last_epoch"] = epoch + 1
        parsed = parse_mask_run_label(run_label)
        if parsed.get("trial") is not None:
            grp.attrs["trial"] = int(parsed["trial"])
        if parsed.get("subset_size") is not None:
            grp.attrs["subset_size"] = int(parsed["subset_size"])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def run_offline(cfg: OfflineMotorPccConfig) -> Dict:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tm_checkpoint = Path(cfg.tm_checkpoint_path)
    h5_path = cfg.output_h5_path

    summary: Dict = {
        "subset_runs": {},
        "motor_location_mm": cfg.motor_position_mm,
        "num_random_trials": cfg.num_random_trials,
        "offline": True,
        "focus_metric_definition": PBR_DEFINITION,
    }
    geometry = load_slm_geometry(cfg)
    if geometry is None:
        print(
            "SLM geometry files not found (PM_center_x/y.npy). "
            "Saving 128x128 phase masks only; pack to full SLM in the lab."
        )
    else:
        print("SLM geometry loaded; saving both small and full (1200x1920) phase masks.")

    tm_system = initialize_tm_system(cfg)
    n_modes = len(tm_system.guided_modes)
    subset_sizes = cfg.pcc_subset_sizes or _subset_sizes_for_n_modes(n_modes)
    n_samples = cfg.n_samples if cfg.n_samples > 0 else 5 * n_modes
    max_subset = max(subset_sizes)
    if n_samples < max_subset:
        raise ValueError(f"Need at least {max_subset} samples, got {n_samples}")

    summary["n_guided_modes"] = n_modes
    summary["n_samples"] = n_samples
    summary["subset_sizes"] = subset_sizes
    print(f"Guided modes N={n_modes} | dataset size={n_samples}")

    et_ratio = cfg.transmission_image_et[0] / cfg.transmission_fourier_et[0]
    bs_split = 0.47736
    magnification = 50.0
    tran_im_factor = magnification**2 * bs_split / (1 - bs_split) * et_ratio

    if not cfg.skip_tm_optimization:
        print("Step 1/3: Initial TM retrieval (Adam) from initial_calibration (both planes)")
        train_data, test_data = load_real_mmf_data(
            h5_file_path=h5_path,
            experiment_group=cfg.initial_h5_group,
            train_test_split_ratio=cfg.train_test_split_ratio,
            random_seed=cfg.random_seed,
            im_factor=tran_im_factor,
        )
        train_data["slm_patterns"] = train_data["slm_patterns"].to(tm_system.device)
        train_data["intensity_targets"] = train_data["intensity_targets"].to(tm_system.device)
        test_data["slm_patterns"] = test_data["slm_patterns"].to(tm_system.device)
        test_data["intensity_targets"] = test_data["intensity_targets"].to(tm_system.device)

        if cfg.use_tm_checkpoint:
            if not tm_checkpoint.exists():
                raise FileNotFoundError(
                    f"TM checkpoint not found at {tm_checkpoint}. "
                    "The published release does not include the checkpoint; "
                    "omit --use-tm-checkpoint so step 1 (Adam joint calibration) regenerates it."
                )
            print(f"Loading TM checkpoint from {tm_checkpoint}")
            tm_system = torch.load(tm_checkpoint, map_location=tm_system.device, weights_only=False)
            initial_tm_history = {"loaded_from_checkpoint": True}
        else:
            initial_tm_history = optimize_tm_slm(
                tm_system=tm_system,
                slm_patterns=train_data["slm_patterns"],
                intensity_targets=train_data["intensity_targets"],
                num_epochs=cfg.tm_num_epochs,
                learning_rate=cfg.tm_learning_rate,
                batch_size=cfg.tm_batch_size,
                planes="both",
                optimize_aberrations=cfg.optimize_aberrations,
                test_slm_patterns=test_data["slm_patterns"],
                test_intensity_targets=test_data["intensity_targets"],
            )
            tm_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tm_system, tm_checkpoint)
        summary["initial_tm_history_keys"] = sorted(initial_tm_history.keys())
    else:
        if not tm_checkpoint.exists():
            raise FileNotFoundError(
                f"--skip-tm-optimization requires checkpoint at {tm_checkpoint}. "
                "The published release does not include the checkpoint; "
                "run step 1 (omit --skip-tm-optimization) to regenerate it."
            )
        print(f"Loading TM checkpoint from {tm_checkpoint}")
        tm_system = torch.load(tm_checkpoint, map_location=tm_system.device, weights_only=False)
        summary["initial_tm_history_keys"] = ["loaded_from_checkpoint"]

    print("Step 2/3: Calibrating TM system (both planes) + reinitializing TM for PCC")
    tm_system.calibrate()
    summary["step1_optimize_planes"] = "both"
    summary["step2_calibrate_planes"] = "both"
    pcc_tm_init = tm_system.optimized_tm.detach().clone()
    validation = tm_system.validate_aberrated_mode_basis()
    if not validation.get("ok"):
        raise RuntimeError(f"Aberrated mode basis validation failed: {validation}")

    if cfg.skip_pcc:
        print("Skipping PCC benchmarks (--skip-pcc)")
        summary["config"] = asdict(cfg)
        summary_path = output_dir / "motor_pcc_benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    method = cfg.optimization_method.lower()
    if method != "admm":
        raise ValueError(
            f"optimization_method must be 'admm', got {cfg.optimization_method!r}"
        )
    optimizer_fn = optimize_tm_admm_dual_plane
    summary["optimization_method"] = method
    summary["step3_optimize_planes"] = cfg.pcc_optimize_planes
    if cfg.focus_eval_plane_override in ("image", "fourier"):
        cfg.focus_eval_plane = cfg.focus_eval_plane_override
    else:
        cfg.focus_eval_plane = _focus_eval_plane_for_pcc_planes(cfg.pcc_optimize_planes)
    summary["focus_eval_plane"] = cfg.focus_eval_plane
    summary["tm_checkpoint_path"] = str(cfg.tm_checkpoint_path)

    single_plane_full_set = (
        _pcc_uses_single_plane(cfg.pcc_optimize_planes) and not cfg.single_plane_subset_sweep
    )
    step3_label = "full-set benchmark" if single_plane_full_set else "subset benchmarks"
    print(
        f"Step 3/3: {method.upper()} {step3_label} + saving SLM focus masks "
        f"(TM planes={cfg.pcc_optimize_planes}, focus eval={cfg.focus_eval_plane})"
    )
    post_motor_data = load_all_mmf_data(h5_path, cfg.post_motor_h5_group, tran_im_factor)
    post_motor_data["slm_patterns"] = post_motor_data["slm_patterns"].to(tm_system.device)
    post_motor_data["intensity_targets"] = post_motor_data["intensity_targets"].to(tm_system.device)

    print("Precomputing input modes for post-motor dataset")
    post_motor_input_modes = compute_input_modes(
        tm_system, post_motor_data["slm_patterns"], cfg.input_modes_batch_size
    )
    n_available = post_motor_input_modes.shape[1]
    if single_plane_full_set:
        subset_sizes_step3 = [n_available]
        num_trials_step3 = 1
        summary["pcc_single_plane_full_set"] = True
        summary["num_random_trials"] = 1
        summary["subset_sizes"] = subset_sizes_step3
        print(
            f"Single-plane mode ({cfg.pcc_optimize_planes}): optimizing on all "
            f"{n_available} post-motor measurements (1 run, no subset sweep)"
        )
    else:
        subset_sizes_step3 = subset_sizes
        num_trials_step3 = cfg.num_random_trials
        summary["pcc_single_plane_full_set"] = False
        if n_available < max_subset:
            raise ValueError(
                f"Post-motor group '{cfg.post_motor_h5_group}' has {n_available} samples; "
                f"need at least {max_subset} for the largest subset."
            )

    mask_writer = SlmMaskWriter(
        h5_path=Path(cfg.slm_masks_h5_path),
        geometry=geometry,
        focus_eval_plane=cfg.focus_eval_plane,
    )

    try:
        for subset_size in subset_sizes_step3:
            subset_summary: Dict[str, Any] = {"trials": {}}
            for trial_idx in range(num_trials_step3):
                run_label = mask_run_label(method, subset_size, trial_idx)
                if single_plane_full_set:
                    sample_indices = np.arange(n_available, dtype=np.int64)
                else:
                    sample_indices = sample_random_subset_indices(
                        n_available, subset_size, trial_idx, cfg.random_seed
                    )
                index_tensor = torch.as_tensor(sample_indices, dtype=torch.long, device=tm_system.device)
                if single_plane_full_set:
                    print(f"--- {method.upper()} run: full set ({n_available} measurements) ---")
                else:
                    print(
                        f"--- {method.upper()} run: subset_size={subset_size}, "
                        f"trial={trial_idx + 1}/{num_trials_step3} ---"
                    )
                if str(tm_system.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    tm_system.optimized_tm_tensor = pcc_tm_init.detach().clone()

                simulated_focus_metrics: List[float] = []

                def on_epoch_end(epoch: int, epoch_seconds: float, system: TMMeasurementSystem) -> None:
                    with torch.enable_grad():
                        optimized_phase, focus_metric = optimize_slm_pattern_with_tm(
                            tm_system=system,
                            slm_shape=(cfg.n_out, cfg.n_out),
                            num_iterations=cfg.slm_opt_iters,
                            lr=cfg.slm_opt_lr,
                            center_radius=cfg.focus_center_radius_px,
                            target_plane=cfg.focus_eval_plane,
                        )
                    phase_small_uint8 = phase_to_uint8(optimized_phase.detach().cpu().numpy())
                    mask_writer.append(run_label, epoch, phase_small_uint8, focus_metric)
                    simulated_focus_metrics.append(float(focus_metric))
                    print(
                        f"[{run_label}] epoch {epoch + 1}/{cfg.pcc_num_epochs} | "
                        f"time={epoch_seconds:.2f}s | simulated_PBR={focus_metric:.2f}"
                    )

                input_modes_slice = post_motor_input_modes[:, index_tensor]
                targets_slice = post_motor_data["intensity_targets"][index_tensor]

                optimizer_kwargs: Dict[str, Any] = {
                    "num_epochs": cfg.pcc_num_epochs,
                    "resync_interval": cfg.pcc_resync_interval,
                    "on_epoch_end": on_epoch_end,
                    "planes": cfg.pcc_optimize_planes,
                    "init_method": cfg.admm_init_method,
                    "init_power_iters": cfg.admm_init_power_iters,
                    "init_num_pixels": cfg.admm_init_num_pixels,
                    "init_eig_rel_threshold": cfg.admm_init_eig_rel_threshold,
                }

                run_history = optimizer_fn(
                    tm_system,
                    input_modes_slice,
                    targets_slice,
                    **optimizer_kwargs,
                )
                subset_summary["trials"][str(trial_idx)] = {
                    "run_label": run_label,
                    "sample_indices": sample_indices.tolist(),
                    "epoch_losses": run_history["epoch_losses"],
                    "epoch_times_sec": run_history.get("epoch_times", []),
                    "init_method": run_history.get("init_method"),
                    "init_time_sec": run_history.get("init_seconds"),
                    "init_loss": run_history.get("init_loss"),
                    "simulated_focus_metrics": simulated_focus_metrics,
                }
            summary["subset_runs"][str(subset_size)] = subset_summary
    finally:
        mask_writer.close()

    summary["slm_masks_h5_path"] = str(cfg.slm_masks_h5_path)
    summary["config"] = asdict(cfg)
    summary_path = output_dir / "motor_pcc_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    table_paths = save_results_tables(summary, output_dir)
    summary["table_paths"] = {k: str(v) for k, v in table_paths.items()}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Summary: {summary_path}")
    print(f"SLM masks: {cfg.slm_masks_h5_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline modal TM retrieval from saved H5 measurements (Adam calibration + ADMM)"
    )
    parser.add_argument("--h5-path", type=str, default="data/measurement_motor_pcc.h5")
    parser.add_argument("--output-dir", type=str, default="results/offline/modal")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--use-tm-checkpoint",
        default=False,
        action="store_true",
        help="Load an existing tm_system_checkpoint.pt instead of running step 1. "
        "The checkpoint is not published; omit this flag so step 1 regenerates it.",
    )
    parser.add_argument("--skip-tm-optimization", action="store_true")
    parser.add_argument("--skip-pcc", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="admm",
        choices=["admm"],
        help="TM retrieval for the subset benchmark (only 'admm' is published)",
    )
    parser.add_argument(
        "--admm-init-method",
        type=str,
        default="default_beta_prior",
        choices=["default_beta_prior", "random", "identity"],
    )
    parser.add_argument("--admm-init-power-iters", type=int, default=50)
    parser.add_argument("--admm-init-num-pixels", type=int, default=500)
    parser.add_argument("--admm-init-eig-rel-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--pcc-subset-sizes",
        type=str,
        default="",
        help="Comma-separated subset sizes (default: 1.2N, 1.5N, 2N, 3N)",
    )
    parser.add_argument("--pcc-num-epochs", type=int, default=90)
    parser.add_argument("--tm-num-epochs", type=int, default=100)
    parser.add_argument("--pm-center-x", type=str, default="PM_center_x.npy")
    parser.add_argument("--pm-center-y", type=str, default="PM_center_y.npy")
    parser.add_argument(
        "--num-random-trials",
        type=int,
        default=DEFAULT_NUM_RANDOM_TRIALS,
        help="Number of random subset index draws per subset size (default: 10)",
    )
    parser.add_argument(
        "--step3-mode",
        type=str,
        default="pcc-focus",
        choices=["pcc-focus"],
        help="Step 3: ADMM subset benchmark + per-epoch focus masks",
    )
    parser.add_argument(
        "--pcc-optimize-planes",
        type=str,
        default="both",
        choices=["image", "fourier", "both"],
        help="Step 3 only: measurement planes used for TM retrieval "
        "(image, fourier, or both). Steps 1-2 always optimize/calibrate over both planes. "
        "With image or fourier here, all post-motor measurements are used in a single run "
        "(no subset sweep or random trials). Focus SLM masks maximize focus on the "
        "complementary plane (image TM → Fourier focus; Fourier TM → image focus; both → image).",
    )
    parser.add_argument(
        "--focus-eval-plane",
        type=str,
        default="auto",
        choices=["auto", "image", "fourier"],
        help="Step 3 focus plane. auto: complementary plane for a single-plane TM, image for both.",
    )
    parser.add_argument(
        "--single-plane-subset-sweep",
        action="store_true",
        help="With --pcc-optimize-planes image|fourier, run the subset sweep and random trials "
        "(same budgets as the dual-plane benchmark) instead of one full-set run.",
    )
    parser.add_argument(
        "--tm-checkpoint",
        type=str,
        default="",
        help="Calibrated TM checkpoint to save (step 1) or load (--skip-tm-optimization / "
        "--use-tm-checkpoint). Default: <output-dir>/tm_system_checkpoint.pt",
    )
    args = parser.parse_args()

    cfg = OfflineMotorPccConfig()
    cfg.output_h5_path = args.h5_path
    cfg.output_dir = args.output_dir
    cfg.tm_checkpoint_path = args.tm_checkpoint or str(Path(args.output_dir) / "tm_system_checkpoint.pt")
    cfg.slm_masks_h5_path = str(Path(args.output_dir) / "slm_focus_masks.h5")
    cfg.device = args.device
    cfg.use_tm_checkpoint = args.use_tm_checkpoint
    cfg.skip_tm_optimization = args.skip_tm_optimization
    cfg.skip_pcc = args.skip_pcc
    cfg.optimization_method = args.method
    cfg.admm_init_method = args.admm_init_method
    cfg.admm_init_power_iters = args.admm_init_power_iters
    cfg.admm_init_num_pixels = args.admm_init_num_pixels
    cfg.admm_init_eig_rel_threshold = args.admm_init_eig_rel_threshold
    cfg.pcc_num_epochs = args.pcc_num_epochs
    cfg.tm_num_epochs = args.tm_num_epochs
    cfg.pm_center_x_path = args.pm_center_x
    cfg.pm_center_y_path = args.pm_center_y
    if args.pcc_subset_sizes.strip():
        cfg.pcc_subset_sizes = [int(x.strip()) for x in args.pcc_subset_sizes.split(",") if x.strip()]
    cfg.num_random_trials = args.num_random_trials
    cfg.step3_mode = "pcc_focus"
    cfg.pcc_optimize_planes = args.pcc_optimize_planes
    cfg.focus_eval_plane_override = args.focus_eval_plane
    cfg.single_plane_subset_sweep = args.single_plane_subset_sweep
    cfg.focus_eval_plane = _focus_eval_plane_for_pcc_planes(cfg.pcc_optimize_planes)

    run_offline(cfg)


if __name__ == "__main__":
    main()
