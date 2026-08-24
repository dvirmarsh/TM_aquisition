"""
Project saved SLM focus masks from offline PCC runs and optionally capture
measured center-focus metrics in the lab.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))            # this repository's core modules
sys.path.insert(0, str(REPO_ROOT.parent))     # the private lab repo providing create_refs

from create_refs import get_camera_exposure_minimum, set_camera_pixel_format
from motor_pos_utils import mask_run_sort_key, parse_mask_run_label
from run_wavefront_control import PipelineConfig, TMFocusPipeline
from run_modal_tm_retrieval import _rows_to_markdown_table

TRANSMISSION_CAMERA_INDEX = 0
MEASUREMENT_PIXEL_FORMAT = "Mono12"
DEFAULT_PIXEL_FORMAT = "Mono8"


def _subset_sort_key(subset_key: str) -> Tuple[int, int, str]:
    return mask_run_sort_key(subset_key)


def _parse_subset_key(subset_key: str) -> Dict[str, Any]:
    return parse_mask_run_label(subset_key)


def _safe_index(values: List[Any], idx: int, default: Any = None) -> Any:
    if not values or idx < 0 or idx >= len(values):
        return default
    return values[idx]


def list_subset_keys(
    h5f: h5py.File,
    *,
    subset_key: Optional[str] = None,
    subset_prefix: Optional[str] = None,
) -> List[str]:
    if subset_key is not None:
        if subset_key not in h5f:
            raise KeyError(f"Group '{subset_key}' not found. Available: {list(h5f.keys())}")
        return [subset_key]

    keys = sorted(h5f.keys(), key=_subset_sort_key)
    if subset_prefix:
        keys = [key for key in keys if key.startswith(subset_prefix)]
    if not keys:
        raise ValueError("No subset groups matched the requested filters.")
    return keys


def _image_for_display(image: np.ndarray) -> np.ndarray:
    """Map Mono12 (0-4095) or Mono8 frames to 8-bit PNG preview."""
    if image.max() > 255:
        return np.clip(image / 16.0, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def load_simulated_focus_metrics(h5f: h5py.File, subset_key: str) -> List[float]:
    grp = h5f[subset_key]
    if "simulated_focus_metrics" not in grp:
        return []
    return [float(v) for v in grp["simulated_focus_metrics"][:]]


def replay_masks(
    h5f: h5py.File,
    subset_key: str,
    output_dir: Path,
    pipeline: TMFocusPipeline,
    *,
    start_epoch: int = 1,
    end_epoch: Optional[int] = None,
    settle_time: float = 0.06,
    use_full_masks: bool = True,
    save_artifacts: bool = True,
) -> List[float]:
    measured_metrics: List[float] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    grp = h5f[subset_key]
    if use_full_masks and "phase_masks_full_uint8" in grp:
        masks = grp["phase_masks_full_uint8"]
        mask_kind = "full"
    else:
        masks = grp["phase_masks_small_uint8"]
        mask_kind = "small"

    n_epochs = int(masks.shape[0])
    end = end_epoch if end_epoch is not None else n_epochs
    end = min(end, n_epochs)
    start = max(1, start_epoch)
    print(f"Projecting {mask_kind} masks for {subset_key}, epochs {start}-{end} of {n_epochs}")

    for epoch_idx in range(start - 1, end):
        if mask_kind == "full":
            phase_mask = np.array(masks[epoch_idx], dtype=np.uint8)
        else:
            small = np.array(masks[epoch_idx], dtype=np.uint8)
            phase_mask = pipeline.pack_phase_to_full_slm(small)

        pipeline.project_phase_mask_to_slm(phase_mask)
        time.sleep(settle_time)

        transmission_cam = pipeline.cameras[TRANSMISSION_CAMERA_INDEX]
        min_exposure = get_camera_exposure_minimum(transmission_cam)
        raw = pipeline.capture_hdr_images(transmission_cam, [min_exposure])
        raw = np.squeeze(np.array(raw))
        proc = pipeline.crop_and_resize_image(raw, "transmission_image").astype(np.float32)
        proc = np.flip(proc)
        metric = pipeline.center_background_focus_metric(
            proc,
            center_radius=pipeline.cfg.measured_focus_window,
            target_yx=None,
        )
        measured_metrics.append(float(metric))
        if save_artifacts:
            tag = f"{subset_key}_epoch_{epoch_idx + 1:03d}"
            cv2.imwrite(str(output_dir / f"im_center_focus_{tag}.png"), _image_for_display(proc))
            np.save(output_dir / f"projected_full_phase_mask_uint8_{tag}.npy", phase_mask)
        print(f"  epoch {epoch_idx + 1}: measured_focus={metric:.4f}")

    if save_artifacts:
        out_path = output_dir / f"{subset_key}_measured_focus_metrics.npy"
        np.save(out_path, np.array(measured_metrics, dtype=np.float32))
        print(f"Saved {len(measured_metrics)} metrics to {out_path}")
    return measured_metrics


def build_lab_epoch_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_modes = summary.get("n_guided_modes")
    rows: List[Dict[str, Any]] = []
    subset_runs = summary.get("subset_runs") or {}
    for subset_key in sorted(subset_runs.keys(), key=_subset_sort_key):
        run = subset_runs[subset_key]
        parsed = _parse_subset_key(subset_key)
        subset_size = parsed["subset_size"]
        trial_idx = parsed.get("trial")
        if trial_idx is None:
            trial_idx = run.get("trial")
        simulated = [float(v) for v in (run.get("simulated_focus_metrics") or [])]
        measured = [float(v) for v in (run.get("measured_focus_metrics") or [])]
        epoch_times = [float(v) for v in (run.get("epoch_times_sec") or [])]
        epoch_losses = [float(v) for v in (run.get("epoch_losses") or [])]
        n_epochs = max(len(simulated), len(measured), len(epoch_times), len(epoch_losses))
        cumulative = 0.0
        for epoch_idx in range(n_epochs):
            epoch_time = float(_safe_index(epoch_times, epoch_idx, 0.0) or 0.0)
            cumulative += epoch_time
            sim_focus = _safe_index(simulated, epoch_idx, float("nan"))
            meas_focus = _safe_index(measured, epoch_idx, float("nan"))
            row: Dict[str, Any] = {
                "subset_key": subset_key,
                "run_type": parsed["run_type"],
                "epoch": epoch_idx + 1,
                "simulated_center_focus_metric": round(float(sim_focus), 6),
                "measured_center_focus_metric": round(float(meas_focus), 6),
            }
            if n_modes is not None:
                row["n_guided_modes"] = n_modes
            if trial_idx is not None:
                row["trial"] = int(trial_idx)
            if subset_size is not None:
                row["subset_size"] = subset_size
                if n_modes:
                    row["subset_size_over_N"] = round(subset_size / n_modes, 4)
            if epoch_times:
                row["epoch_time_sec"] = round(epoch_time, 4)
                row["cumulative_epoch_time_sec"] = round(cumulative, 4)
            if epoch_losses:
                row["pcc_loss_1_minus_pcc"] = round(float(_safe_index(epoch_losses, epoch_idx, float("nan"))), 6)
            if simulated and measured and epoch_idx < len(simulated) and epoch_idx < len(measured):
                row["measured_minus_simulated_focus"] = round(float(meas_focus) - float(sim_focus), 6)
            rows.append(row)
    return rows


def build_lab_summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_modes = summary.get("n_guided_modes")
    rows: List[Dict[str, Any]] = []
    subset_runs = summary.get("subset_runs") or {}
    for subset_key in sorted(subset_runs.keys(), key=_subset_sort_key):
        run = subset_runs[subset_key]
        parsed = _parse_subset_key(subset_key)
        subset_size = parsed["subset_size"]
        trial_idx = parsed.get("trial")
        if trial_idx is None:
            trial_idx = run.get("trial")
        simulated = [float(v) for v in (run.get("simulated_focus_metrics") or [])]
        measured = [float(v) for v in (run.get("measured_focus_metrics") or [])]
        epoch_times = [float(v) for v in (run.get("epoch_times_sec") or [])]
        epoch_losses = [float(v) for v in (run.get("epoch_losses") or [])]
        row: Dict[str, Any] = {
            "subset_key": subset_key,
            "run_type": parsed["run_type"],
            "n_epochs": max(len(simulated), len(measured), len(epoch_times), len(epoch_losses)),
            "initial_simulated_focus": round(simulated[0], 6) if simulated else None,
            "final_simulated_focus": round(simulated[-1], 6) if simulated else None,
            "best_simulated_focus": round(max(simulated), 6) if simulated else None,
            "initial_measured_focus": round(measured[0], 6) if measured else None,
            "final_measured_focus": round(measured[-1], 6) if measured else None,
            "best_measured_focus": round(max(measured), 6) if measured else None,
        }
        if n_modes is not None:
            row["n_guided_modes"] = n_modes
        if trial_idx is not None:
            row["trial"] = int(trial_idx)
        if subset_size is not None:
            row["subset_size"] = subset_size
            if n_modes:
                row["subset_size_over_N"] = round(subset_size / n_modes, 4)
        if epoch_times:
            row["total_epoch_time_sec"] = round(sum(epoch_times), 4)
            row["mean_epoch_time_sec"] = round(float(np.mean(epoch_times)), 4)
        if epoch_losses:
            row["initial_pcc_loss"] = round(epoch_losses[0], 6)
            row["final_pcc_loss"] = round(epoch_losses[-1], 6)
            row["best_pcc_loss"] = round(min(epoch_losses), 6)
        if simulated and measured:
            paired = [m - s for m, s in zip(measured, simulated)]
            row["mean_measured_minus_simulated_focus"] = round(float(np.mean(paired)), 6)
            row["final_measured_minus_simulated_focus"] = round(measured[-1] - simulated[-1], 6)
        rows.append(row)
    return rows


def save_lab_results_tables(summary: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = build_lab_epoch_rows(summary)
    summary_rows = build_lab_summary_rows(summary)
    paths = {
        "epoch_csv": output_dir / "lab_replay_epoch_table.csv",
        "summary_csv": output_dir / "lab_replay_subset_summary.csv",
        "combined_md": output_dir / "lab_replay_tables.md",
    }
    for path, rows in ((paths["epoch_csv"], epoch_rows), (paths["summary_csv"], summary_rows)):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    paths["combined_md"].write_text(
        "# Lab replay of SLM focus masks\n\n"
        + _rows_to_markdown_table(summary_rows)
        + "\n\n## Per-epoch (first 25)\n\n"
        + _rows_to_markdown_table(epoch_rows, max_rows=25),
        encoding="utf-8",
    )
    print("\n=== Lab replay subset summary ===")
    print(_rows_to_markdown_table(summary_rows))
    return paths


def _merge_offline_subset_metadata(summary: Dict[str, Any], offline_summary: Dict[str, Any]) -> None:
    summary["n_guided_modes"] = offline_summary.get("n_guided_modes", summary.get("n_guided_modes"))
    summary["motor_location_mm"] = offline_summary.get("motor_location_mm", summary.get("motor_location_mm"))
    offline_subset_runs = offline_summary.get("subset_runs") or {}

    for subset_key, run in summary["subset_runs"].items():
        parsed = _parse_subset_key(subset_key)
        subset_size = parsed.get("subset_size")
        trial_idx = parsed.get("trial")

        offline_run = None
        if subset_size is not None and trial_idx is not None:
            trials = (offline_subset_runs.get(str(subset_size)) or {}).get("trials") or {}
            offline_run = trials.get(str(trial_idx))
        if offline_run is None and subset_size is not None:
            legacy = offline_subset_runs.get(str(subset_size))
            if legacy and "trials" not in legacy:
                offline_run = legacy
        if offline_run is None:
            continue
        for field in ("epoch_times_sec", "epoch_losses", "simulated_focus_metrics"):
            if field not in run or not run[field]:
                run[field] = offline_run.get(field, [])
        if trial_idx is not None:
            run["trial"] = int(trial_idx)


def replay_all_masks(
    masks_h5_path: Path,
    output_dir: Path,
    *,
    subset_key: Optional[str] = None,
    subset_prefix: Optional[str] = None,
    offline_summary_path: Optional[Path] = None,
    start_epoch: int = 1,
    end_epoch: Optional[int] = None,
    settle_time: float = 0.06,
    use_full_masks: bool = True,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"subset_runs": {}, "masks_h5_path": str(masks_h5_path)}
    if offline_summary_path is not None and offline_summary_path.exists():
        with open(offline_summary_path, encoding="utf-8") as f:
            offline_summary = json.load(f)
        _merge_offline_subset_metadata(summary, offline_summary)

    pipeline = TMFocusPipeline(PipelineConfig())
    pipeline.setup()
    transmission_cam = pipeline.cameras[TRANSMISSION_CAMERA_INDEX]
    try:
        set_camera_pixel_format(transmission_cam, MEASUREMENT_PIXEL_FORMAT)
        with h5py.File(masks_h5_path, "r") as h5f:
            subset_keys = list_subset_keys(h5f, subset_key=subset_key, subset_prefix=subset_prefix)
            print(f"Replaying {len(subset_keys)} subset group(s): {subset_keys}")
            for key in subset_keys:
                print(f"\n--- {key} ---")
                measured_metrics = replay_masks(
                    h5f,
                    key,
                    output_dir,
                    pipeline,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    settle_time=settle_time,
                    use_full_masks=use_full_masks,
                    save_artifacts=save_artifacts,
                )
                simulated_metrics = load_simulated_focus_metrics(h5f, key)
                parsed = _parse_subset_key(key)
                run_data: Dict[str, Any] = {
                    "measured_focus_metrics": measured_metrics,
                    "simulated_focus_metrics": simulated_metrics[: len(measured_metrics)],
                }
                if parsed.get("trial") is not None:
                    run_data["trial"] = int(parsed["trial"])
                if key in summary["subset_runs"]:
                    summary["subset_runs"][key].update(run_data)
                else:
                    summary["subset_runs"][key] = run_data
    finally:
        set_camera_pixel_format(transmission_cam, DEFAULT_PIXEL_FORMAT)
        pipeline.teardown()

    table_paths = save_lab_results_tables(summary, output_dir)
    summary["table_paths"] = {k: str(v) for k, v in table_paths.items()}
    summary_path = output_dir / "lab_replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Project saved SLM focus masks in the lab")
    parser.add_argument(
        "--masks-h5",
        type=str,
        default="results/offline/modal/slm_focus_masks.h5",
        help="H5 file written by run_modal_tm_retrieval.py",
    )
    parser.add_argument(
        "--subset-key",
        type=str,
        default="",
        help="Replay a single H5 group (default: replay all groups in the file)",
    )
    parser.add_argument(
        "--subset-prefix",
        type=str,
        default="",
        help="Only replay groups with this prefix, e.g. pcc_subset_",
    )
    parser.add_argument(
        "--offline-summary-json",
        type=str,
        default="",
        help="Optional motor_pcc_benchmark_summary.json for PCC loss / TM error columns",
    )
    parser.add_argument("--output-dir", type=str, default="results/lab_replay")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=0, help="0 = through last epoch")
    parser.add_argument("--settle-time", type=float, default=0.06)
    parser.add_argument(
        "--small-masks-only",
        action="store_true",
        help="Pack 128x128 masks with local geometry instead of using saved full masks",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Skip saving per-epoch PNG/NPY artifacts (metrics and tables are still saved)",
    )
    args = parser.parse_args()

    end_epoch = args.end_epoch if args.end_epoch > 0 else None
    subset_key = args.subset_key.strip() or None
    subset_prefix = args.subset_prefix.strip() or None
    offline_summary_path = Path(args.offline_summary_json) if args.offline_summary_json.strip() else None
    if offline_summary_path is None:
        default_summary = Path(args.output_dir).parent / "motor_pcc_benchmark_summary.json"
        if default_summary.exists():
            offline_summary_path = default_summary

    replay_all_masks(
        Path(args.masks_h5),
        Path(args.output_dir),
        subset_key=subset_key,
        subset_prefix=subset_prefix,
        offline_summary_path=offline_summary_path,
        start_epoch=args.start_epoch,
        end_epoch=end_epoch,
        settle_time=args.settle_time,
        use_full_masks=not args.small_masks_only,
        save_artifacts=not args.no_save_artifacts,
    )


if __name__ == "__main__":
    main()
