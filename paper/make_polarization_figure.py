"""Supplementary Fig. S1: two foci behind the rotating polarizer, and the axes of the camera weights.

Panels (a), (b): one SLM pattern designed with the retrieved TM alone (camera weight b in the
forward model, no polarizer information), recorded behind the polarizer at stage 6 and 96 deg.
Panel (c): angle between the two polarizations of five pairs of orthogonal, linearly polarized foci
behind the unequal weights, against the designed orientation of the pair.

Inputs: results/polarization/figure_frames/ (camera counts, 16-bit PNG) and the table "Pairs" of
results/polarization/polarizer_followup_20261115_1342/followup_report.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMES = REPO_ROOT / "results" / "polarization" / "figure_frames"
OUTPUT = Path(__file__).resolve().parent / "figures" / "polarization_axes_experiment.png"

EXPOSURE = 192                       # microseconds; no saturated pixel at either angle
PANELS = [
    ("pattern_a_analyzer_006deg_stage_006deg_exposure_%04d_scan_up_counts.png" % EXPOSURE, r"Polarizer at $6^\circ$", "(a)"),
    ("pattern_a_analyzer_096deg_stage_096deg_exposure_%04d_scan_up_counts.png" % EXPOSURE, r"Polarizer at $96^\circ$", "(b)"),
]
# Pairs of foci, linear and orthogonal in the frame of the TM (follow-up report, 15 Nov 2026):
# designed orientation phi, measured angle between the two polarizations, uncertainty (deg)
PAIRS = np.array([[-4, 106.0, 0.6], [0, 97.0, 0.4], [4, 89.4, 0.5], [8, 80.3, 0.5], [12, 72.9, 0.5]])
BLUE = "#2a78d6"


def read_counts(path: Path) -> np.ndarray:
    """16-bit PNG whose pixel value is the camera count."""
    img = plt.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    return np.round(img.astype(np.float64) * 65535.0)


def main() -> None:
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, (name, label, tag) in zip(axes[:2], PANELS):
        img = read_counts(FRAMES / name)
        ax.imshow(img, cmap="gray", vmin=0, vmax=img.max(), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tag, fontweight="bold", fontsize=14)
        ax.set_xlabel(label, fontsize=12)

    ax = axes[2]
    phi, angle, err = PAIRS.T
    slope, intercept = np.polyfit(phi, angle, 1)
    phi0 = (90.0 - intercept) / slope
    grid = np.linspace(-6, 14, 50)
    ax.axhline(90.0, color="#52514e", lw=1)
    ax.plot(grid, slope * grid + intercept, color=BLUE, lw=2, zorder=2)
    ax.errorbar(phi, angle, yerr=err, fmt="o", ms=7, color=BLUE, mec="white", mew=1.5, ecolor=BLUE, elinewidth=1.2, zorder=3)
    ax.plot([phi0], [90.0], "o", ms=11, mfc="none", mec="black", mew=1.2, zorder=4)
    ax.annotate(r"$90^\circ$ at $\varphi=%.1f^\circ$" % phi0, xy=(phi0, 90.0), xytext=(phi0 - 9.2, 79.0), fontsize=11,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.set_xlim(-6, 14); ax.set_ylim(65, 112)
    ax.set_xticks([-4, 0, 4, 8, 12])
    ax.set_xlabel(r"Designed orientation $\varphi$ of the pair (deg)")
    ax.set_ylabel("Angle between the\ntwo polarizations (deg)")
    ax.grid(color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("(c)", fontweight="bold", fontsize=14)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT} (fit: angle = {slope:.3f} phi + {intercept:.2f}; 90 deg at phi = {phi0:.2f} deg)")


if __name__ == "__main__":
    main()
