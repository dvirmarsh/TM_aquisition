"""
Plot measured lab focus metrics against accumulated offline epoch time.

For each subset size, aggregates across random trials (mean ± std) and plots
one curve per algorithm.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SUBSET_SIZES = (1809, 2262, 3016, 4524)
SUBSET_SIZE_LABELS = {
    1809: "1.2M",
    2262: "1.5M",
    3016: "2M",
    4524: "3M",
}

AlgorithmSeries = Tuple[str, Path, Path, str]


def _trial_from_row(row: dict) -> int:
    raw = row.get("trial", "")
    if raw in ("", "None", None):
        return 0
    return int(raw)


def _load_lab_measured_focus(path: Path) -> Dict[Tuple[int, int, int], float]:
    """Return {(subset_size, epoch, trial): measured_center_focus_metric}."""
    out: Dict[Tuple[int, int, int], float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["subset_size"]), int(row["epoch"]), _trial_from_row(row))
            out[key] = float(row["measured_center_focus_metric"])
    return out


def _load_benchmark_cumulative_time(path: Path) -> Dict[Tuple[int, int, int], float]:
    """Return {(subset_size, epoch, trial): cumulative_epoch_time_sec}."""
    out: Dict[Tuple[int, int, int], float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["subset_size"]), int(row["epoch"]), _trial_from_row(row))
            out[key] = float(row["cumulative_epoch_time_sec"])
    return out


def _subplot_grid(n_panels: int) -> Tuple[int, int]:
    if n_panels <= 1:
        return 1, 1
    if n_panels == 2:
        return 1, 2
    if n_panels <= 4:
        return 2, 2
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def _detect_overlapping_subset_sizes(
    measured: Dict[Tuple[int, int, int], float],
    cumulative: Dict[Tuple[int, int, int], float],
) -> List[int]:
    sizes = {
        size
        for (size, epoch, trial) in measured
        if (size, epoch, trial) in cumulative
    }
    if not sizes:
        raise ValueError("No overlapping (subset_size, epoch, trial) keys between lab and benchmark CSVs")
    return sorted(sizes)


def _aggregate_by_subset_epoch(
    measured: Dict[Tuple[int, int, int], float],
    cumulative: Dict[Tuple[int, int, int], float],
    subset_size: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Return (mean_times, mean_focus, std_focus) sorted by epoch."""
    by_epoch: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for (size, epoch, trial), focus in measured.items():
        if size != subset_size:
            continue
        time_key = (size, epoch, trial)
        if time_key not in cumulative:
            continue
        by_epoch[epoch].append((cumulative[time_key], focus))

    if not by_epoch:
        raise ValueError(f"No overlapping epochs for subset_size={subset_size}")

    epochs = sorted(by_epoch)
    mean_times: List[float] = []
    mean_focus: List[float] = []
    std_focus: List[float] = []
    for epoch in epochs:
        pairs = by_epoch[epoch]
        times = [p[0] for p in pairs]
        focus_vals = [p[1] for p in pairs]
        mean_times.append(float(np.mean(times)))
        mean_focus.append(float(np.mean(focus_vals)))
        std_focus.append(float(np.std(focus_vals, ddof=0)) if len(focus_vals) > 1 else 0.0)
    return mean_times, mean_focus, std_focus


def plot_lab_focus_vs_time(
    algorithms: Sequence[AlgorithmSeries],
    *,
    subset_sizes: Sequence[int] = DEFAULT_SUBSET_SIZES,
    output_path: Path,
    suptitle: str | None = None,
    show: bool = False,
) -> None:
    if not subset_sizes:
        raise ValueError("subset_sizes must not be empty")

    nrows, ncols = _subplot_grid(len(subset_sizes))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    focus_threshold = 700.0

    for ax, subset_size in zip(axes_flat, subset_sizes):
        final_times: List[float] = []
        for label, lab_path, benchmark_path, color in algorithms:
            measured = _load_lab_measured_focus(lab_path)
            cumulative = _load_benchmark_cumulative_time(benchmark_path)
            times, focus_mean, focus_std = _aggregate_by_subset_epoch(
                measured, cumulative, subset_size
            )
            times_arr = np.array(times)
            mean_arr = np.array(focus_mean)
            std_arr = np.array(focus_std)
            ax.plot(times_arr, mean_arr, label=label, color=color, linewidth=1.8)
            ax.fill_between(
                times_arr,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            if len(times_arr) > 0:
                final_times.append(float(times_arr[-1]))
            # Vertical dashed line at first crossing of the focus threshold.
            above = np.where(mean_arr >= focus_threshold)[0]
            if len(above) > 0:
                t_cross = float(times_arr[above[0]])
                ax.axvline(
                    t_cross,
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                )
                ax.text(
                    t_cross,
                    0.5,
                    f"{t_cross:.2f}s",
                    transform=ax.get_xaxis_transform(),
                    color=color,
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                    clip_on=True,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                    },
                )

        if final_times:
            ax.set_xlim(left=0.0, right=min(final_times))
        size_label = SUBSET_SIZE_LABELS.get(subset_size, str(subset_size))
        ax.set_title(f"Subset size = {size_label}")
        ax.set_xlabel("time (seconds)")
        ax.set_ylabel("Measured center focus metric (mean ± std)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fourier_plane_lab_focus_vs_time(
    lab_path: Path,
    benchmark_path: Path,
    *,
    output_path: Path,
    label: str = "Fourier plane",
    color: str = "#1f77b4",
    show: bool = False,
) -> None:
    measured = _load_lab_measured_focus(lab_path)
    cumulative = _load_benchmark_cumulative_time(benchmark_path)
    subset_sizes = _detect_overlapping_subset_sizes(measured, cumulative)
    plot_lab_focus_vs_time(
        [(label, lab_path, benchmark_path, color)],
        subset_sizes=subset_sizes,
        output_path=output_path,
        suptitle=(
            "Fourier plane: lab measured focus vs. accumulated offline epoch time "
            "(mean ± std over trials)"
        ),
        show=show,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot lab measured focus vs accumulated offline epoch time (mean±std over trials)."
    )
    parser.add_argument(
        "--pcc-lab-csv",
        type=Path,
        default=project_root / "results" / "lab_replay" / "modal_replay_epoch_table.csv",
    )
    parser.add_argument(
        "--pcc-benchmark-csv",
        type=Path,
        default=project_root / "results" / "offline" / "modal" / "motor_pcc_benchmark_epoch_table.csv",
    )
    parser.add_argument(
        "--raf-lab-csv",
        type=Path,
        default=project_root / "results" / "lab_replay" / "raf21_lab_replay_epoch_table.csv",
    )
    parser.add_argument(
        "--raf-benchmark-csv",
        type=Path,
        default=project_root / "results" / "offline" / "raf21" / "raf_benchmark_epoch_table.csv",
    )
    parser.add_argument(
        "--admm-lab-csv",
        type=Path,
        default=project_root / "results" / "lab_replay" / "admm_lab_replay_epoch_table.csv",
    )
    parser.add_argument(
        "--admm-benchmark-csv",
        type=Path,
        default=project_root / "results" / "offline" / "admm" / "admm_benchmark_epoch_table.csv",
    )
    parser.add_argument(
        "--subset-sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SUBSET_SIZES),
        help="Comma-separated subset sizes (must be 4 values for 2x2 layout).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "paper" / "figures" / "lab_focus_vs_accumulated_time.png",
    )
    parser.add_argument(
        "--fourier-plane-lab-csv",
        type=Path,
        default=project_root / "results" / "lab_replay" / "lab_replay_epoch_table_fourier_plane.csv",
    )
    parser.add_argument(
        "--fourier-plane-benchmark-csv",
        type=Path,
        default=project_root
        / "results"
        / "offline"
        / "modal"
        / "motor_pcc_benchmark_epoch_table_fourier_plane.csv",
    )
    parser.add_argument(
        "--fourier-plane-output",
        type=Path,
        default=project_root / "paper" / "figures" / "lab_focus_vs_accumulated_time_fourier_plane.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    subset_sizes = [int(x.strip()) for x in args.subset_sizes.split(",") if x.strip()]
    algorithms: List[AlgorithmSeries] = [
        ("Modal", args.pcc_lab_csv, args.pcc_benchmark_csv, "#1f77b4"),
        ("RAF21", args.raf_lab_csv, args.raf_benchmark_csv, "#ff7f0e"),
        ("ADMM", args.admm_lab_csv, args.admm_benchmark_csv, "#2ca02c"),
    ]

    plot_lab_focus_vs_time(
        algorithms,
        subset_sizes=subset_sizes,
        output_path=args.output,
        show=args.show,
    )
    plot_fourier_plane_lab_focus_vs_time(
        args.fourier_plane_lab_csv,
        args.fourier_plane_benchmark_csv,
        output_path=args.fourier_plane_output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
