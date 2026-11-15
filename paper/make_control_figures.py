"""Generate annotated wavefront-control figures for the manuscript."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PAPER_DIR / "image_results"
CONTROL_OUTPUT_PATH = PAPER_DIR / "figures" / "tm_wavefront_control.png"
CROSS_PLANE_OUTPUT_PATH = PAPER_DIR / "figures" / "tm_cross_plane_focus.png"
VECTORIAL_OUTPUT_PATH = PAPER_DIR / "figures" / "tm_vectorial_control.png"
MODAL_OUTPUT_PATH = PAPER_DIR / "figures" / "tm_modal_control.png"

RESULT_ROWS = [
    [
        ("im_image_single_spot.png", r"$\mathrm{PBR}=521.1$"),
        ("im_multispot_image.png", r"$\mathrm{PBR}_3=158.2$"),
        ("im_target_corr_target_1_image.png", r"$\mathrm{PCC}=0.926$"),
        ("im_target_corr_target_2_image.png", r"$\mathrm{PCC}=0.922$"),
    ],
    [
        ("im_fourier_single_spot.png", r"$\mathrm{PBR}=525.3$"),
        ("im_multispot_fourier.png", r"$\mathrm{PBR}_3=180.6$"),
        ("im_target_corr_target_1_fourier.png", r"$\mathrm{PCC}=0.857$"),
        ("im_target_corr_target_2_fourier.png", r"$\mathrm{PCC}=0.879$"),
    ],
]

# Selective excitation of one guided mode: a single SLM pattern, recorded in both planes.
# PCC with the TM-predicted intensity of the target mode (mode_profile_pcc in run_summary.json).
MODAL_PANELS = [
    ("im_one_hot_modes_image_he811.png", r"$\mathrm{PCC}=0.891$", "Image plane", "(a)"),
    ("im_one_hot_modes_fourier_he811.png", r"$\mathrm{PCC}=0.895$", "Fourier plane", "(b)"),
]

TARGET_IMAGES = [
    "target_canvas_target_1_image.png",
    "target_canvas_target_2_image.png",
]

CROSS_PLANE_PANELS = [
    (
        "image from fourier.png",
        r"$\mathrm{PBR}=316.7$",
        "Image plane\n(TM from Fourier)",
        "(a)",
    ),
    (
        "fourier_from_image.png",
        r"$\mathrm{PBR}=433.3$",
        "Fourier plane\n(TM from image)",
        "(b)",
    ),
]

VECTORIAL_PANELS = [
    ("no_pol_img.png", "No polarizer", "(a)"),
    ("x_pol_img.png", r"$x'$ polarizer", "(b)"),
    ("y_pol_img.png", r"$y'$ polarizer", "(c)"),
]


def _load_raw_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return np.maximum(image.astype(np.float64), 0.0)


def load_for_display(path: Path) -> np.ndarray:
    """Scale a raw camera image linearly to [0, 1] without peak clipping/saturation."""
    image = _load_raw_image(path)
    upper = float(image.max()) if image.size else 0.0
    if upper <= 0.0:
        return np.zeros_like(image)

    # Normalize by the true maximum so bright pixels are never saturated.
    return image / upper


def style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_metric_overlay(axis: plt.Axes, metric: str) -> None:
    axis.text(
        0.035,
        0.955,
        metric,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9.5,
        bbox={
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.62,
            "pad": 1.6,
        },
    )


def generate_control_figure() -> None:
    CONTROL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 4, figsize=(9.0, 4.9), facecolor="white")

    for result_row in range(2):
        for column, (filename, metric) in enumerate(RESULT_ROWS[result_row]):
            axis = axes[result_row, column]
            axis.imshow(
                load_for_display(IMAGE_DIR / filename),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            add_metric_overlay(axis, metric)
            style_image_axis(axis)

            if column >= 2:
                target_axis = inset_axes(
                    axis,
                    width="24%",
                    height="24%",
                    loc="lower right",
                    borderpad=0.0,
                )
                target_axis.imshow(
                    load_for_display(IMAGE_DIR / TARGET_IMAGES[column - 2]),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                target_axis.set_xticks([])
                target_axis.set_yticks([])
                for spine in target_axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("red")
                    spine.set_linewidth(1.8)

    figure.subplots_adjust(
        left=0.168,
        right=0.995,
        top=0.875,
        bottom=0.035,
        wspace=0.055,
        hspace=0.055,
    )

    figure.text(
        0.031,
        0.625,
        "Image\nplane",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.031,
        0.265,
        "Fourier\nplane",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    main_centers = [
        axes[0, column].get_position().x0 + axes[0, column].get_position().width / 2
        for column in range(4)
    ]
    for x_position, label in zip(main_centers[:2], ("(a)", "(b)")):
        figure.text(
            x_position,
            0.92,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    figure.text(
        (main_centers[2] + main_centers[3]) / 2,
        0.92,
        "(c)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    figure.savefig(
        CONTROL_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def generate_cross_plane_figure() -> None:
    CROSS_PLANE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), facecolor="white")

    for axis, (filename, metric, subtitle, label) in zip(axes, CROSS_PLANE_PANELS):
        axis.imshow(
            load_for_display(IMAGE_DIR / filename),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        add_metric_overlay(axis, metric)
        style_image_axis(axis)
        axis.set_xlabel(subtitle, fontsize=11, labelpad=8)
        axis.set_title(label, fontsize=14, fontweight="bold", pad=8)

    figure.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.18, wspace=0.08)
    figure.savefig(
        CROSS_PLANE_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def generate_modal_figure() -> None:
    MODAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), facecolor="white")

    for axis, (filename, metric, subtitle, label) in zip(axes, MODAL_PANELS):
        axis.imshow(
            load_for_display(IMAGE_DIR / filename),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        if metric:
            add_metric_overlay(axis, metric)
        style_image_axis(axis)
        axis.set_xlabel(subtitle, fontsize=11, labelpad=8)
        axis.set_title(label, fontsize=14, fontweight="bold", pad=8)

    figure.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.18, wspace=0.08)
    figure.savefig(
        MODAL_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def generate_vectorial_figure() -> None:
    VECTORIAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), facecolor="white")

    for axis, (filename, subtitle, label) in zip(axes, VECTORIAL_PANELS):
        axis.imshow(
            load_for_display(IMAGE_DIR / filename),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        style_image_axis(axis)
        axis.set_xlabel(subtitle, fontsize=12, labelpad=8)
        axis.set_title(label, fontsize=14, fontweight="bold", pad=8)

    figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.18, wspace=0.06)
    figure.savefig(
        VECTORIAL_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    generate_control_figure()
    generate_modal_figure()
    generate_cross_plane_figure()
    generate_vectorial_figure()


if __name__ == "__main__":
    main()
