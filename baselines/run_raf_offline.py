"""
Offline RAF2-1 benchmark: retrieve the pixel-basis TM from saved lab measurements,
loop over probe-count subsets, optimize SLM focus masks after each RAF epoch, and save
masks for later lab projection (no hardware required).

Uses image-plane intensities only (transmission_image); Fourier-plane data is ignored.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raf21_tm import RAF21Options, raf21_tm_epochs
from metrics_losses import PBR_DEFINITION, _focus_metric_from_intensity_map, peak_to_background_ratio
from motor_pos_utils import (
    DEFAULT_NUM_RANDOM_TRIALS,
    mask_run_label,
    parse_mask_run_label,
    sample_random_subset_indices,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _subset_sizes_for_n_slm_pixels(n_slm_pixels: int) -> List[int]:
    return [int(1.2 * 1508), int(1.5 * 1508), 2 * 1508, 3 * 1508]


def _safe_index(values: List[Any], idx: int, default: Any = None) -> Any:
    if not values or idx < 0 or idx >= len(values):
        return default
    return values[idx]


def _iter_trial_subset_runs(summary: Dict) -> List[Tuple[int, int, Dict]]:
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
    n_slm_pixels = summary.get("n_slm_pixels") or 1
    rows: List[Dict[str, Any]] = []
    for subset_size, trial_idx, run in _iter_trial_subset_runs(summary):
        epoch_times = run.get("epoch_times_sec") or []
        ls_losses = run.get("ls_loss") or []
        focus_metrics = run.get("simulated_focus_metrics") or []
        n_epochs = max(len(epoch_times), len(ls_losses), len(focus_metrics))
        cumulative = 0.0
        for epoch_idx in range(n_epochs):
            epoch_time = float(_safe_index(epoch_times, epoch_idx, 0.0) or 0.0)
            cumulative += epoch_time
            rows.append(
                {
                    "n_slm_pixels": n_slm_pixels,
                    "subset_size": subset_size,
                    "trial": trial_idx,
                    "subset_size_over_N": round(subset_size / n_slm_pixels, 4),
                    "epoch": epoch_idx + 1,
                    "epoch_time_sec": round(epoch_time, 4),
                    "cumulative_epoch_time_sec": round(cumulative, 4),
                    "raf_ls_loss": round(float(_safe_index(ls_losses, epoch_idx, float("nan"))), 6),
                    "simulated_center_focus_metric": round(
                        float(_safe_index(focus_metrics, epoch_idx, float("nan"))), 6
                    ),
                }
            )
    return rows


def build_subset_summary_rows(summary: Dict) -> List[Dict[str, Any]]:
    n_slm_pixels = summary.get("n_slm_pixels") or 1
    rows: List[Dict[str, Any]] = []
    for subset_size, trial_idx, run in _iter_trial_subset_runs(summary):
        epoch_times = [float(v) for v in (run.get("epoch_times_sec") or [])]
        ls_losses = [float(v) for v in (run.get("ls_loss") or [])]
        focus_metrics = [float(v) for v in (run.get("simulated_focus_metrics") or [])]
        rows.append(
            {
                "n_slm_pixels": n_slm_pixels,
                "subset_size": subset_size,
                "trial": trial_idx,
                "subset_size_over_N": round(subset_size / n_slm_pixels, 4),
                "n_epochs": max(len(epoch_times), len(ls_losses), len(focus_metrics)),
                "total_epoch_time_sec": round(sum(epoch_times), 4) if epoch_times else None,
                "mean_epoch_time_sec": round(float(np.mean(epoch_times)), 4) if epoch_times else None,
                "initial_raf_ls_loss": round(ls_losses[0], 6) if ls_losses else None,
                "final_raf_ls_loss": round(ls_losses[-1], 6) if ls_losses else None,
                "best_raf_ls_loss": round(min(ls_losses), 6) if ls_losses else None,
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


def save_results_tables(
    summary: Dict,
    output_dir: Path,
    name_prefix: str = "raf_benchmark",
    title: str = "RAF",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = build_epoch_results_rows(summary)
    summary_rows = build_subset_summary_rows(summary)
    paths = {
        "epoch_csv": output_dir / f"{name_prefix}_epoch_table.csv",
        "summary_csv": output_dir / f"{name_prefix}_subset_summary.csv",
        "combined_md": output_dir / f"{name_prefix}_tables.md",
    }
    for path, rows in ((paths["epoch_csv"], epoch_rows), (paths["summary_csv"], summary_rows)):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    paths["combined_md"].write_text(
        f"# {title} offline benchmark results\n\n"
        + _rows_to_markdown_table(summary_rows)
        + "\n\n## Per-epoch (first 25)\n\n"
        + _rows_to_markdown_table(epoch_rows, max_rows=25),
        encoding="utf-8",
    )
    print(f"\n=== {title} subset summary ===")
    print(_rows_to_markdown_table(summary_rows))
    return paths


def phase_to_uint8(phase_rad: np.ndarray) -> np.ndarray:
    phase_wrapped = np.mod(phase_rad, 2.0 * np.pi)
    return np.floor(phase_wrapped * (256.0 / (2.0 * np.pi))).astype(np.uint8)


def load_slm_geometry(cfg: "OfflineRafConfig") -> Optional[Dict]:
    pm_x_path = Path(cfg.pm_center_x_path)
    pm_y_path = Path(cfg.pm_center_y_path)
    if not pm_x_path.exists() or not pm_y_path.exists():
        return None
    return {
        "pm_center_x": int(np.load(pm_x_path).item()),
        "pm_center_y": int(np.load(pm_y_path).item()),
        "slm_ratio": cfg.slm_ratio,
        "slm_n_out": cfg.slm_n_out,
    }


def pack_raf_phase_to_full_slm(phase_uint8_small: np.ndarray, geometry: Dict) -> np.ndarray:
    """Pack a 32x32 macro-grid phase mask to the full 1200x1920 SLM aperture."""
    ratio = geometry["slm_ratio"]
    n = geometry["slm_n_out"]
    pm_center_x = geometry["pm_center_x"]
    pm_center_y = geometry["pm_center_y"]

    small = np.zeros((1200 // ratio, 1920 // ratio), dtype=np.uint8)
    cy = round(pm_center_y / ratio)
    cx = round(pm_center_x / ratio)
    y_start = cy - n // 2
    x_start = cx - n // 2
    y_end = y_start + n
    x_end = x_start + n
    h = min(n, phase_uint8_small.shape[0], small.shape[0] - y_start)
    w = min(n, phase_uint8_small.shape[1], small.shape[1] - x_start)
    if y_start < 0 or x_start < 0 or y_end > small.shape[0] or x_end > small.shape[1]:
        raise ValueError(
            f"SLM macro crop ({y_start}:{y_end}, {x_start}:{x_end}) is out of bounds "
            f"for downsampled SLM shape {small.shape}"
        )
    small[y_start : y_start + h, x_start : x_start + w] = np.flip(phase_uint8_small[:h, :w])
    return np.kron(small, np.ones((ratio, ratio), dtype=np.uint8))


def build_raf_tensors(
    slm_phase: torch.Tensor,
    intensity_image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert lab arrays to RAF2-1 tensors.

    Args:
        slm_phase: (P, n_pix_slm, n_pix_slm) radians.
        intensity_image: (P, n_pix_out, n_pix_out) image-plane intensities.

    Returns:
        y_intensity: (n_pix_out^2, P)
        x_probes: (n_pix_slm^2, P)
    """
    p = slm_phase.shape[0]
    amp = intensity_image.clamp(min=0).sqrt()
    amp_max = amp.max().clamp(min=1e-30)
    amp_norm = amp / amp_max
    y_intensity = (amp_norm**2).reshape(p, -1).T.contiguous()
    x_probes = torch.exp(1j * slm_phase.to(dtype=torch.float32)).reshape(p, -1).T.contiguous().to(torch.complex64)
    return y_intensity, x_probes


def load_raf_h5_data(
    h5_path: str,
    experiment_group: str,
    *,
    n_samples: Optional[int] = None,
    im_factor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    print(f"Loading RAF data from {h5_path}, group: {experiment_group}")
    with h5py.File(h5_path, "r") as f:
        if experiment_group not in f:
            raise ValueError(f"Group '{experiment_group}' not found. Available: {list(f.keys())}")
        group = f[experiment_group]
        if "transmission_image" not in group or "PhaseMask" not in group:
            raise ValueError("H5 group must contain transmission_image and PhaseMask datasets.")
        slm_phase = np.flip(np.array(group["PhaseMask"][:]), axis=(-2, -1))
        intensity_image = np.flip(np.array(group["transmission_image"][:]), axis=(-2, -1)).astype(np.float32)
    if n_samples is not None:
        slm_phase = slm_phase[:n_samples]
        intensity_image = intensity_image[:n_samples]
    if slm_phase.dtype == np.uint8:
        slm_phase = slm_phase.astype(np.float32) * (2 * np.pi / 256.0)
    intensity_image *= im_factor
    return {
        "slm_phase": torch.from_numpy(slm_phase).float(),
        "intensity_image": torch.from_numpy(intensity_image).float(),
    }


def optimize_raf_focus_mask(
    a_pixel: torch.Tensor,
    n_pix_slm: int,
    n_pix_out: int,
    num_iterations: int,
    lr: float,
    center_radius: int,
) -> Tuple[torch.Tensor, float]:
    """
    Optimize SLM phase against the retrieved pixel-basis TM (maximizing the centre-window
    enhancement) and return the phase with the PBR of its predicted intensity
    (metrics_losses.PBR_DEFINITION).
    """
    device = a_pixel.device
    a_pixel_detached = a_pixel.detach()
    slm_phase_low = nn.Parameter(2 * torch.pi * torch.rand((n_pix_slm, n_pix_slm), dtype=torch.float32, device=device))
    optimizer = optim.Adam([slm_phase_low], lr=lr)
    with torch.enable_grad():
        for _ in range(num_iterations):
            optimizer.zero_grad()
            x_flat = torch.exp(1j * slm_phase_low).reshape(-1)
            e_out = a_pixel_detached @ x_flat
            i_pred = (e_out.abs() ** 2).reshape(n_pix_out, n_pix_out)
            focus_metric = _focus_metric_from_intensity_map(i_pred, center_radius)
            (-focus_metric).backward()
            optimizer.step()
    with torch.no_grad():
        phi_low = torch.remainder(slm_phase_low.detach(), 2 * torch.pi)
        x_flat = torch.exp(1j * phi_low).reshape(-1)
        e_out = a_pixel_detached @ x_flat
        i_pred = (e_out.abs() ** 2).reshape(n_pix_out, n_pix_out)
    final_pbr, _ = peak_to_background_ratio(i_pred)
    return phi_low, final_pbr


class RafSlmMaskWriter:
    def __init__(
        self,
        h5_path: Path,
        geometry: Optional[Dict],
        slm_n_out: int,
        focus_eval_plane: str = "image",
    ):
        self.h5_path = Path(h5_path)
        self.geometry = geometry
        self.slm_n_out = slm_n_out
        self.focus_eval_plane = focus_eval_plane
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[h5py.File] = None
        self._reset_labels: set = set()

    def _ensure_group(self, run_label: str) -> h5py.Group:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "a")
        if run_label not in self._reset_labels:
            if run_label in self._file:
                del self._file[run_label]
            self._reset_labels.add(run_label)
        if run_label not in self._file:
            grp = self._file.create_group(run_label)
            grp.create_dataset(
                "phase_masks_small_uint8",
                shape=(0, self.slm_n_out, self.slm_n_out),
                maxshape=(None, self.slm_n_out, self.slm_n_out),
                dtype=np.uint8,
                chunks=(1, self.slm_n_out, self.slm_n_out),
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
            full_mask = pack_raf_phase_to_full_slm(phase_small_uint8, self.geometry)
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


@dataclass
class OfflineRafConfig:
    slm_n_out: int = 32
    n_out: int = 128
    slm_active_extent_px: int = 640
    transmission_image_et: Tuple[float, ...] = (120.0,)
    random_seed: int = 42
    device: str = "cuda"
    raf_iters: int = 200
    # ratio/beta/gamma/npower_iter are the values of the reference RAF21_tm_cpu.m (beta, gamma
    # and npower_iter are fixed constants there); only mu is tuned.
    raf_mu: float = 0.6
    raf_ratio: float = 2.0 / 3.0
    raf_beta: float = 5.0
    raf_gamma: float = 0.5
    raf_npower_iter: int = 30
    slm_opt_iters: int = 150
    slm_opt_lr: float = 0.1
    focus_center_radius_px: int = 1
    n_samples: int = 0
    h5_group: str = "experiment_0"
    num_random_trials: int = DEFAULT_NUM_RANDOM_TRIALS
    raf_subset_sizes: Optional[List[int]] = None
    output_dir: str = "results/offline/raf21"
    output_h5_path: str = "data/measurement_raf.h5"
    slm_masks_h5_path: str = "results/offline/raf21/slm_focus_masks.h5"
    pm_center_x_path: str = "PM_center_x.npy"
    pm_center_y_path: str = "PM_center_y.npy"

    @property
    def slm_ratio(self) -> int:
        return self.slm_active_extent_px // self.slm_n_out


def run_offline(cfg: OfflineRafConfig) -> Dict:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = cfg.device
    if runtime_device == "cuda" and not torch.cuda.is_available():
        runtime_device = "cpu"

    n_slm_pixels = cfg.slm_n_out * cfg.slm_n_out
    subset_sizes = cfg.raf_subset_sizes or _subset_sizes_for_n_slm_pixels(n_slm_pixels)
    max_subset = max(subset_sizes)

    with h5py.File(cfg.output_h5_path, "r") as f:
        if cfg.h5_group not in f:
            raise ValueError(f"Group '{cfg.h5_group}' not found in {cfg.output_h5_path}")
        n_available = int(f[cfg.h5_group]["PhaseMask"].shape[0])
    n_samples = cfg.n_samples if cfg.n_samples > 0 else n_available
    if n_samples < max_subset:
        raise ValueError(f"Need at least {max_subset} samples, got {n_samples} (available {n_available})")

    summary: Dict = {
        "subset_runs": {},
        "num_random_trials": cfg.num_random_trials,
        "offline": True,
        "focus_metric_definition": PBR_DEFINITION,
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

    im_factor = 1.0
    full_data = load_raf_h5_data(cfg.output_h5_path, cfg.h5_group, n_samples=n_samples, im_factor=im_factor)
    full_data["slm_phase"] = full_data["slm_phase"].to(runtime_device)
    full_data["intensity_image"] = full_data["intensity_image"].to(runtime_device)

    mask_writer = RafSlmMaskWriter(
        h5_path=Path(cfg.slm_masks_h5_path),
        geometry=geometry,
        slm_n_out=cfg.slm_n_out,
    )

    opts = RAF21Options(
        mu=cfg.raf_mu,
        iters=cfg.raf_iters,
        ratio=cfg.raf_ratio,
        beta=cfg.raf_beta,
        gamma=cfg.raf_gamma,
        npower_iter=cfg.raf_npower_iter,
    )
    try:
        for subset_size in subset_sizes:
            subset_summary: Dict[str, Any] = {"trials": {}}
            for trial_idx in range(cfg.num_random_trials):
                run_label = mask_run_label("raf", subset_size, trial_idx)
                sample_indices = sample_random_subset_indices(
                    n_samples, subset_size, trial_idx, cfg.random_seed
                )
                index_tensor = torch.as_tensor(sample_indices, dtype=torch.long, device=runtime_device)
                print(
                    f"--- RAF run: subset_size={subset_size}, "
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
                        f"[{run_label}] epoch {epoch + 1}/{cfg.raf_iters} | "
                        f"time={epoch_seconds:.2f}s | ls_loss={ls_loss:.4f} | simulated_PBR={focus_metric:.2f}"
                    )

                raf21_tm_epochs(
                    y_intensity,
                    x_probes,
                    opts,
                    generator=gen,
                    on_epoch_end=on_epoch_end,
                )
                subset_summary["trials"][str(trial_idx)] = {
                    "run_label": run_label,
                    "sample_indices": sample_indices.tolist(),
                    "ls_loss": ls_losses_acc,
                    "epoch_times_sec": epoch_times_acc,
                    "simulated_focus_metrics": simulated_focus_metrics,
                }
            summary["subset_runs"][str(subset_size)] = subset_summary
    finally:
        mask_writer.close()

    summary["slm_masks_h5_path"] = str(cfg.slm_masks_h5_path)
    summary["config"] = asdict(cfg)
    summary_path = output_dir / "raf_benchmark_summary.json"
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
    parser = argparse.ArgumentParser(description="Offline RAF2-1 benchmark from saved H5 measurements")
    parser.add_argument("--h5-path", type=str, default="data/measurement_raf.h5")
    parser.add_argument("--h5-group", type=str, default="experiment_0")
    parser.add_argument("--output-dir", type=str, default="results/offline/raf21")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--raf-iters", type=int, default=200)
    parser.add_argument("--raf-mu", type=float, default=0.4)
    parser.add_argument("--raf-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--raf-beta", type=float, default=5.0)
    parser.add_argument("--raf-gamma", type=float, default=0.5)
    parser.add_argument("--raf-npower-iter", type=int, default=30)
    parser.add_argument("--slm-opt-iters", type=int, default=150)
    parser.add_argument("--slm-opt-lr", type=float, default=0.1)
    parser.add_argument("--focus-center-radius", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=0, help="0 = use all samples in the H5 group")
    parser.add_argument(
        "--raf-subset-sizes",
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

    cfg = OfflineRafConfig()
    cfg.output_h5_path = args.h5_path
    cfg.h5_group = args.h5_group
    cfg.output_dir = args.output_dir
    cfg.slm_masks_h5_path = str(Path(args.output_dir) / "slm_focus_masks.h5")
    cfg.device = args.device
    cfg.raf_iters = args.raf_iters
    cfg.raf_mu = args.raf_mu
    cfg.raf_ratio = args.raf_ratio
    cfg.raf_beta = args.raf_beta
    cfg.raf_gamma = args.raf_gamma
    cfg.raf_npower_iter = args.raf_npower_iter
    cfg.slm_opt_iters = args.slm_opt_iters
    cfg.slm_opt_lr = args.slm_opt_lr
    cfg.focus_center_radius_px = args.focus_center_radius
    cfg.n_samples = args.n_samples
    cfg.slm_n_out = args.slm_n_out
    cfg.n_out = args.n_out
    cfg.pm_center_x_path = args.pm_center_x
    cfg.pm_center_y_path = args.pm_center_y
    if args.raf_subset_sizes.strip():
        cfg.raf_subset_sizes = [int(x.strip()) for x in args.raf_subset_sizes.split(",") if x.strip()]
    cfg.num_random_trials = args.num_random_trials

    run_offline(cfg)


if __name__ == "__main__":
    main()
