"""
Collect the images of the SLM pattern with the best separation of two foci along the two polarization axes:
the polarizer along x, the polarizer along y, and no polarizer, in one folder.

    python collect_best_two_foci_images.py --prepare      # the two images already recorded + the SLM pattern (no hardware)
    python collect_best_two_foci_images.py --capture      # record the image WITHOUT polarizer (SLM + Fourier camera), add it

The pattern is test (b) of the follow-up of 15 Nov 2026 with lab x at stage 6 deg (tag x6_lab_axes_U_from_foci_v2 in
tm_focus_outputs/polarizer_followup_20261115_1342): focus A along lab x (stage 6 deg), focus B along lab y (stage 96 deg).
The images behind the polarizer were cropped from the camera frame around the centre displaced by the polarizer, so
they are in the geometry of the image without polarizer (standard crop of the pipeline).

Folder: tm_focus_outputs/best_two_foci_three_images/. Every image is saved as raw counts (.npy, 128 x 128, float32)
and as a 16-bit PNG; the PNGs of the three images share ONE scale. The rotation stage is not used by --capture.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import h5py
import numpy as np

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_polarization_axes as rpa            # noqa: E402
import run_polarizer_followup as rpf           # noqa: E402
import run_polarizer_test as rpt               # noqa: E402

SOURCE = TM_LIB / "tm_focus_outputs" / "polarizer_followup_20261115_1342"
TAG = "x6_lab_axes_U_from_foci_v2"
OUT = TM_LIB / "tm_focus_outputs" / "best_two_foci_three_images"
NAMES = {"x": "image_polarizer_along_x_stage_6deg", "y": "image_polarizer_along_y_stage_96deg", "none": "image_no_polarizer"}
EXPOSURE = 24.0


def window_info(img: np.ndarray, support: np.ndarray) -> Dict:
    out = {}
    for name, tgt in (("A", rpt.TARGET_A), ("B", rpt.TARGET_B)):
        pk = rpf.locate_peak(img[None], support, tgt)
        out[name] = {"peak_yx": list(pk), "window_power_3x3": float(rpf.window_powers(img[None], support, pk, 1)[0]),
                     "peak_count": float(img[pk]), "pbr": float(rpa.peak_to_background_ratio(img, support, pk)[0])}
    return out


def write_images(out: Path, images: Dict[str, np.ndarray]) -> float:
    """Raw .npy for each image and 16-bit PNGs with one common scale; returns the count that maps to full scale."""
    peak = max(float(im.max()) for im in images.values())
    for key, im in images.items():
        np.save(out / f"{NAMES[key]}.npy", im.astype(np.float32))
        u16 = np.clip(np.round(im.astype(np.float64) * (65535.0 / peak)), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / f"{NAMES[key]}.png"), u16)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        titles = {"x": "polarizer along x (stage 6 deg)", "y": "polarizer along y (stage 96 deg)", "none": "no polarizer"}
        keys = [k for k in ("x", "y", "none") if k in images]
        fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.3), squeeze=False)
        for ax, k in zip(axes[0], keys):
            ax.imshow(images[k], cmap="inferno", vmin=0, vmax=peak)
            for tgt, colour in ((rpt.TARGET_A, "cyan"), (rpt.TARGET_B, "lime")):
                ax.plot(tgt[1], tgt[0], marker="o", ms=16, mfc="none", mec=colour, mew=0.8)
            ax.set_title(titles[k], fontsize=10)
            ax.set_axis_off()
        fig.suptitle("One SLM pattern, one colour scale. Cyan: focus A (x polarized). Green: focus B (y polarized).", fontsize=9)
        fig.tight_layout()
        fig.savefig(str(out / "three_images.png" if len(keys) == 3 else out / "two_images_with_polarizer.png"), dpi=170)
        plt.close(fig)
    except Exception as exc:
        print(f"(figure not saved: {exc})")
    return peak


def load_polarizer_images() -> Dict[str, np.ndarray]:
    z = np.load(SOURCE / f"two_foci_{TAG}_frames.npz")
    ang = [float(a) for a in z["angles_deg"]]
    return {"x": z["frames"][ang.index(0.0)].astype(np.float32), "y": z["frames"][ang.index(90.0)].astype(np.float32)}


def prepare() -> Dict:
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_polarizer_images()
    support = rpt.disk(images["x"].shape[-1], 58.1)
    for suffix, new in (("phase_uint8.npy", "slm_pattern_phase_uint8.npy"), ("full_slm_mask_uint8.npy", "slm_pattern_full_slm_mask_uint8.npy"),
                        ("phase_rad.npy", "slm_pattern_phase_rad.npy"), ("phase_hsv.png", "slm_pattern_phase_hsv.png")):
        shutil.copy2(SOURCE / f"two_foci_{TAG}_{suffix}", OUT / new)
    b = json.load(open(SOURCE / "polarizer_followup_b_summary.json"))["test_b_x_angle"]
    info = {"pattern": f"{SOURCE.name}/two_foci_{TAG}", "what": "focus A along lab x (stage 6 deg), focus B along lab y (stage 96 deg); "
            "U from the foci calibration at stage 6 / 96 deg, weighted TM of 15 Nov 2026",
            "targets_yx": {"A": list(rpt.TARGET_A), "B": list(rpt.TARGET_B)},
            "with_polarizer": {"recorded": b["time"], "minutes_after_the_dataset": b["minutes_since_dataset"], "exposure": b["exposure"],
                               "stage_angles_deg": {"x": 6.0, "y": 96.0}, "pass_fraction_A_at_x": b["transmitted_fraction_A_at_x"],
                               "pass_fraction_B_at_x": b["transmitted_fraction_B_at_x"], "separation_raw": b["separation"],
                               "separation_before_arm": b["before_arm"]["separation"],
                               "foci": {k: window_info(images[k], support) for k in ("x", "y")}},
            "note": "At exposure 24 the camera reads small powers 5 to 9 counts low per 3x3 window, so a blocked focus reads about zero."}
    info["png_full_scale_counts"] = write_images(OUT, images)
    json.dump(info, open(OUT / "info.json", "w"), indent=1)
    print(f"Prepared {OUT}: the two images behind the polarizer and the SLM pattern.")
    for k in ("x", "y"):
        f = info["with_polarizer"]["foci"][k]
        print(f"  polarizer along {k}: focus A {f['A']['window_power_3x3']:.0f} counts at {f['A']['peak_yx']}, focus B {f['B']['window_power_3x3']:.0f} counts at {f['B']['peak_yx']}")
    return info


def capture(rig, data_fresh=None) -> Dict:
    """Record the pattern without polarizer: standard crop of the pipeline, exposure 24 (halved if it saturated and the camera allowed it)."""
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.load(open(OUT / "info.json"))
    images = load_polarizer_images()
    support = rpt.disk(rig.n, 58.1)
    u8 = np.load(OUT / "slm_pattern_phase_uint8.npy")
    rig.show(u8)
    raw = rig.grab_fourier(EXPOSURE)
    saturated = bool(raw.max() >= rpt.SATURATION)
    img = rig.to_grid(rig.crop(raw, (0, 0)))
    images["none"] = img
    foci = window_info(img, support)
    res = {"recorded": datetime.now().isoformat(timespec="seconds"), "exposure": float(max(EXPOSURE, rig.min_exposure)), "saturated": saturated,
           "largest_count_on_the_sensor": float(raw.max()), "foci": foci, "dark_level_outside_disk": float(np.median(img[~support]))}
    # where the foci are, against the images behind the polarizer (registration check)
    px, py = info["with_polarizer"]["foci"]["x"], info["with_polarizer"]["foci"]["y"]
    res["peak_shift_px"] = {"A": [foci["A"]["peak_yx"][0] - px["A"]["peak_yx"][0], foci["A"]["peak_yx"][1] - px["A"]["peak_yx"][1]],
                            "B": [foci["B"]["peak_yx"][0] - py["B"]["peak_yx"][0], foci["B"]["peak_yx"][1] - py["B"]["peak_yx"][1]]}
    res["power_ratio_to_polarizer_image"] = {"A_no_polarizer_over_A_at_x": foci["A"]["window_power_3x3"] / px["A"]["window_power_3x3"],
                                             "B_no_polarizer_over_B_at_y": foci["B"]["window_power_3x3"] / py["B"]["window_power_3x3"]}
    if data_fresh is not None:
        res["freshness"] = data_fresh
    info["without_polarizer"] = res
    info["png_full_scale_counts"] = write_images(OUT, images)
    old = OUT / "two_images_with_polarizer.png"
    if old.exists():
        old.unlink()                                     # replaced by three_images.png (the two single images are kept)
    json.dump(info, open(OUT / "info.json", "w"), indent=1)
    print(f"Image without polarizer: exposure {res['exposure']:g}, largest count on the sensor {raw.max():.0f}, saturated: {saturated}")
    for name in ("A", "B"):
        f = foci[name]
        print(f"  focus {name}: {f['window_power_3x3']:.0f} counts (3x3) at {f['peak_yx']}, peak {f['peak_count']:.0f}, PBR {f['pbr']:.0f}; "
              f"shift against the polarizer image {res['peak_shift_px'][name]} px")
    print(f"  power without polarizer / power behind the polarizer: A {res['power_ratio_to_polarizer_image']['A_no_polarizer_over_A_at_x']:.2f}, "
          f"B {res['power_ratio_to_polarizer_image']['B_no_polarizer_over_B_at_y']:.2f}")
    print(f"All three images are in {OUT}")
    return res


def freshness_without_polarizer(rig, n_masks: int = 8) -> Dict:
    """Correlation of probe frames (no polarizer) with the dataset frames of the same masks, r <= 50 px."""
    with h5py.File(TM_LIB / "measurement_focus.h5", "r") as h5:
        n_total = int(h5["experiment_0"]["transmission_fourier"].shape[0])
        idx = rpt.held_out_indices(n_total)[:n_masks]
        ref = np.asarray(h5["experiment_0"]["transmission_fourier"][idx], dtype=np.float32)
    with h5py.File(rig.pipeline.cfg.optimized_masks_h5_path, "r") as h5:
        masks = np.asarray(h5["mode_excitation_patterns"]["phase_masks_small_uint8"][idx], dtype=np.uint8)
    fit = rpt.disk(rig.n, rpt.FIT_RADIUS)
    pcc, lag, level = [], [], []
    frames = []
    for k in range(n_masks):
        rig.show(masks[k])
        raw = rig.grab_fourier(rig.speckle_exposure)
        frames.append(rig.to_grid(rig.crop(raw, (0, 0))))
    for k in range(n_masks):
        pcc.append(rpa.pcc(frames[k][fit], ref[k][fit]))
        level.append(float(frames[k][fit].mean() / ref[k][fit].mean()))
        if k:
            lag.append(rpa.pcc(frames[k][fit], ref[k - 1][fit]))
    t_dataset = (TM_LIB / "measurement_focus.h5").stat().st_mtime
    res = {"pcc_8_masks": [float(v) for v in pcc], "mean": float(np.mean(pcc)), "mean_against_the_previous_mask": float(np.mean(lag)),
           "level_now_over_dataset": float(np.mean(level)), "minutes_after_the_dataset": (datetime.now().timestamp() - t_dataset) / 60.0}
    print(f"Probe frames without polarizer against the dataset frames: mean correlation {res['mean']:.3f} (previous mask: {res['mean_against_the_previous_mask']:.3f}); "
          f"level now / dataset {res['level_now_over_dataset']:.2f}; {res['minutes_after_the_dataset']:.0f} min after the dataset")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    if args.capture:
        from run_tm_focus_optimization import PipelineConfig, TMFocusPipeline      # hardware libraries are loaded here

        pipeline = TMFocusPipeline(PipelineConfig())
        pipeline.setup()
        try:
            rig = rpf.GuardedRig(pipeline, None, float(pipeline.cameras[1].ExposureTime.GetMin()))     # no stage: it is not moved
            fresh = freshness_without_polarizer(rig)
            capture(rig, fresh)
        finally:
            pipeline.teardown()


if __name__ == "__main__":
    main()
