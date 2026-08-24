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
# Decades of dynamic range for logarithmic focus display.
LOG_DYNAMIC_RANGE = 3.0

RESULT_ROWS = [
    [
        ("im_image_1.png", r"$\eta=835.8$", True),
        ("im_multispot_image.png", r"$\eta_3=186.0$", True),
        ("im_one_hot_modes_image.png", r"$\mathrm{PCC}=0.905$", False),
        ("im_target_corr_target_1_image.png", r"$\mathrm{PCC}=0.931$", False),
        ("im_target_corr_target_2_image.png", r"$\mathrm{PCC}=0.935$", False),
    ],
    [
        ("im_fourier_1.png", r"$\eta=663.5$", True),
        ("im_multispot_fourier.png", r"$\eta_3=238.4$", True),
        ("im_one_hot_modes_fourier.png", r"$\mathrm{PCC}=0.927$", False),
        ("im_target_corr_target_1_fourier.png", r"$\mathrm{PCC}=0.880$", False),
        ("im_target_corr_target_2_fourier.png", r"$\mathrm{PCC}=0.898$", False),
    ],
]

TARGET_IMAGES = [
    "target_canvas_target_1_image.png",
    "target_canvas_target_2_image.png",
]

CROSS_PLANE_PANELS = [
    (
        "image from fourier.png",
        r"$\eta=493.1$",
        "Image plane\n(TM from Fourier)",
        "(a)",
    ),
    (
        "fourier_from_image.png",
        r"$\eta=381.3$",
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


def load_for_display(path: Path, *, logarithmic: bool = False) -> np.ndarray:
    """Scale a raw camera image for display without peak clipping/saturation."""
    image = _load_raw_image(path)
    upper = float(image.max()) if image.size else 0.0
    if upper <= 0.0:
        return np.zeros_like(image)

    # Normalize by the true maximum so bright pixels are never saturated.
    image = image / upper
    if not logarithmic:
        return image

    floor = 10.0 ** (-LOG_DYNAMIC_RANGE)
    log_image = np.log10(image + floor)
    log_min = np.log10(floor)
    log_max = np.log10(1.0 + floor)
    return (log_image - log_min) / (log_max - log_min)


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

    figure, axes = plt.subplots(2, 5, figsize=(11.2, 4.9), facecolor="white")

    for result_row in range(2):
        for column, (filename, metric, logarithmic) in enumerate(RESULT_ROWS[result_row]):
            axis = axes[result_row, column]
            axis.imshow(
                load_for_display(IMAGE_DIR / filename, logarithmic=logarithmic),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            add_metric_overlay(axis, metric)
            style_image_axis(axis)

            if column >= 3:
                target_axis = inset_axes(
                    axis,
                    width="24%",
                    height="24%",
                    loc="lower right",
                    borderpad=0.0,
                )
                target_axis.imshow(
                    load_for_display(IMAGE_DIR / TARGET_IMAGES[column - 3]),
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
        left=0.135,
        right=0.995,
        top=0.875,
        bottom=0.035,
        wspace=0.055,
        hspace=0.055,
    )

    figure.text(
        0.025,
        0.625,
        "Image\nplane",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.025,
        0.265,
        "Fourier\nplane",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    main_centers = [
        axes[0, column].get_position().x0 + axes[0, column].get_position().width / 2
        for column in range(5)
    ]
    for x_position, label in zip(main_centers[:3], ("(a)", "(b)", "(c)")):
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
        (main_centers[3] + main_centers[3]) / 2,
        0.92,
        "(d)",
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
            load_for_display(IMAGE_DIR / filename, logarithmic=True),
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


def generate_vectorial_figure() -> None:
    VECTORIAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), facecolor="white")

    for axis, (filename, subtitle, label) in zip(axes, VECTORIAL_PANELS):
        axis.imshow(
            load_for_display(IMAGE_DIR / filename, logarithmic=True),
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
    generate_cross_plane_figure()
    generate_vectorial_figure()


if __name__ == "__main__":
    main()
