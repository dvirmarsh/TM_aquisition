"""
Find the stage angle at which the rotating polarizer (KDC101 stage, in front of the transmission Fourier-plane
camera) is parallel to a static polarizer placed in front of it.

    python find_static_polarizer_angle.py [--mask-index 0]

Principle: behind the static polarizer the light is linearly polarized along its axis, whatever the fiber does, so
the power behind the rotating polarizer follows Malus's law, P(angle) = P_max cos^2(angle - angle_static) + floor.
The minimum (extinction) is located much more sharply than the maximum, so the script measures the extinction
angle and adds 90 deg.

Procedure
 1. One fixed probe mask on the SLM (any pattern gives the same answer; a speckle fills the NA disk).
 2. Coarse scan, stage 0 .. 170 deg in steps of 10 deg, at an exposure without saturation. Malus fit.
 3. Fine scan, +-12 deg around the coarse extinction angle in steps of 1 deg, upward and then downward, at
    several exposures. The extinction angle is the minimum of c0 + c1 cos(2 angle) + c2 sin(2 angle) fitted to the
    scan, at the longest exposure that does not saturate anywhere in the range.
    If the range would cross stage 0 it is taken 180 deg further (the same analyzer).
 4. Coarse check of the maximum, +-20 deg around extinction + 90 deg in steps of 5 deg.

Power = sum of the camera counts over a disk that contains the NA disk for every stage angle, minus the dark level
(median of the sensor far from the image). One warm-up frame is discarded after every change of stage angle,
SLM pattern or exposure. Output: tm_focus_outputs/static_polarizer_<date>_<time>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_polarization_axes as rpa            # noqa: E402
import run_polarizer_followup as rpf           # noqa: E402
import run_polarizer_test as rpt               # noqa: E402

COARSE = tuple(float(a) for a in range(0, 180, 10))
FINE_HALF, FINE_STEP = 12.0, 1.0
EXPOSURE_START = 675.0
EXPOSURE_FACTORS = (1.0, 4.0, 16.0, 64.0)
EXPOSURE_LIMIT = 200000.0                      # microseconds; keeps a frame well inside the retrieval time-out
DISK_MARGIN_CAM = 110                          # camera pixels added to the radius of the NA disk (25 px of the model grid)
OUTER_GAP_CAM = 60


def power(rig, raw: np.ndarray, centre=None) -> Dict:
    """Dark-subtracted sum over a disk around the image, the largest count in it, and the dark level."""
    yc, xc = (rig.yc, rig.xc) if centre is None else centre
    radius = rig.half * 58.1 / 64.0 + DISK_MARGIN_CAM
    yy, xx = np.ogrid[:raw.shape[0], :raw.shape[1]]
    r2 = (yy - yc) ** 2 + (xx - xc) ** 2
    inside, outside = r2 <= radius ** 2, r2 >= (radius + OUTER_GAP_CAM) ** 2
    dark = float(np.median(raw[outside]))
    return {"power": float((raw[inside].astype(np.float64) - dark).sum()), "max": float(raw[inside].max()), "dark": dark,
            "saturated": bool(raw[inside].max() >= rpt.SATURATION)}


def scan(rig, angles: Sequence[float], exposures: Sequence[float], centre=None) -> Dict:
    out = {e: {"power": [], "max": [], "dark": [], "saturated": []} for e in exposures}
    for a in angles:
        rig.move(a % 360.0)
        for e in exposures:
            p = power(rig, rig.grab_fourier(e), centre)
            for k in out[e]:
                out[e][k].append(p[k])
    return {"angles": [float(a) for a in angles], "by_exposure": out}


def image_centre(rig, raw: np.ndarray):
    """Centroid of the image (camera pixels), to centre the disk if the static polarizer has displaced the image."""
    dark = np.median(raw)
    img = np.clip(raw.astype(np.float64) - dark, 0, None)
    img[img < 0.05 * img.max()] = 0.0
    yy, xx = np.indices(img.shape)
    return float((img * yy).sum() / img.sum()), float((img * xx).sum() / img.sum())


def run(rig, mask_u8: np.ndarray, out_dir: Path, exposure_max: float) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    res: Dict = {"started": datetime.now().isoformat(timespec="seconds")}

    def save() -> None:
        with open(out_dir / "static_polarizer_summary.json", "w", encoding="utf-8") as f:
            json.dump(rpa._to_jsonable(res), f, indent=1)

    rig.show(mask_u8)

    # ---- coarse scan, at an exposure without saturation
    print("\n=== Coarse scan, 0 .. 170 deg in steps of 10 deg ===")
    e0 = max(EXPOSURE_START, rig.min_exposure)
    while True:
        coarse = scan(rig, COARSE, [e0])
        c = coarse["by_exposure"][e0]
        if not any(c["saturated"]) or e0 <= rig.min_exposure:
            break
        e0 = max(e0 / 2.0, rig.min_exposure)
        print(f"  saturation: repeating at exposure {e0:g}")
    ang, p = np.array(coarse["angles"]), np.array(c["power"])
    mf = rpf.malus_fit(ang, p)
    ext0 = (mf["azimuth_deg"] + 90.0) % 180.0
    res["coarse"] = {"exposure": e0, "angles_deg": ang.tolist(), "power": p.tolist(), "max_count": c["max"], "dark": c["dark"], "saturated": bool(any(c["saturated"])),
                     "malus_fit": mf, "extinction_angle_deg": ext0, "minimum_over_maximum_measured": float(p.min() / p.max())}
    print(f"  exposure {e0:g}; power per angle (millions of counts): {[round(v / 1e6, 2) for v in p]}")
    print(f"  Malus fit: maximum at {mf['azimuth_deg']:.1f} deg, extinction at {ext0:.1f} deg; fitted degree of linear polarization {mf['degree_of_linear_polarization']:.3f}; "
          f"measured minimum / maximum {p.min() / p.max():.4f}; largest count {max(c['max']):.0f}")
    # centre of the image at the brightest angle (the static polarizer may displace it)
    rig.move(float(ang[int(np.argmax(p))]) % 360.0)
    cy, cx = image_centre(rig, rig.grab_fourier(e0))
    res["coarse"]["image_centre_cam_px"] = [cy, cx]
    res["coarse"]["image_centre_minus_nominal_cam_px"] = [cy - rig.yc, cx - rig.xc]
    print(f"  centre of the image at the brightest angle: ({cy - rig.yc:+.0f}, {cx - rig.xc:+.0f}) camera px from the nominal centre")
    if np.hypot(cy - rig.yc, cx - rig.xc) > 80.0:
        print("  WARNING: the image is far from the nominal centre; part of it may fall outside the summed disk.")
    save()

    # ---- fine scan around the extinction
    centre = float(np.round(ext0))
    if centre < FINE_HALF + 3.0:
        centre += 180.0                                     # the same analyzer, away from the stage zero
    fine_angles = [centre + k for k in np.arange(-FINE_HALF, FINE_HALF + 0.01, FINE_STEP)]
    exposures = sorted({min(max(e0 * f, rig.min_exposure), exposure_max, EXPOSURE_LIMIT) for f in EXPOSURE_FACTORS})
    print(f"\n=== Fine scan, stage {fine_angles[0]:g} .. {fine_angles[-1]:g} deg in steps of {FINE_STEP:g} deg, exposures {[float(e) for e in exposures]} ===")
    up = scan(rig, fine_angles, exposures)
    down = scan(rig, fine_angles[::-1], exposures)
    fits: Dict = {}
    for e in exposures:
        u, d = up["by_exposure"][e], down["by_exposure"][e]
        if any(u["saturated"]) or any(d["saturated"]):
            print(f"  exposure {e:g}: saturated somewhere in the range, not used")
            continue
        fu = rpf.extinction_fit(up["angles"], u["power"])
        fd = rpf.extinction_fit(down["angles"], d["power"])
        fits[e] = {"up": fu, "down": fd, "power_up": u["power"], "power_down": d["power"], "max_count": float(max(u["max"]))}
        print(f"  exposure {e:g}: extinction upward {fu['extinction_angle_deg']:.3f} deg (fit {fu['sigma_fit_deg']:.3f}, bootstrap {fu['sigma_bootstrap_deg']:.3f}), "
              f"downward {fd['extinction_angle_deg']:.3f} deg; minimum / maximum of the fit {fu['minimum_over_maximum_of_fit']:.5f}; "
              f"residual / amplitude {fu['rms_residual_over_amplitude']:.4f}; largest count {max(u['max']):.0f}")
    if not fits:
        raise RuntimeError("Every exposure saturates in the fine range.")
    best = max(fits)
    fb = fits[best]
    ext = 0.5 * (fb["up"]["extinction_angle_deg"] + fb["down"]["extinction_angle_deg"])
    diff = fb["down"]["extinction_angle_deg"] - fb["up"]["extinction_angle_deg"]
    stat = 0.5 * float(np.hypot(max(fb["up"]["sigma_fit_deg"], fb["up"]["sigma_bootstrap_deg"]), max(fb["down"]["sigma_fit_deg"], fb["down"]["sigma_bootstrap_deg"])))
    all_ext = [0.5 * (v["up"]["extinction_angle_deg"] + v["down"]["extinction_angle_deg"]) for v in fits.values()]
    spread = float(np.ptp(all_ext))
    sigma = float(np.sqrt(stat ** 2 + (0.5 * diff) ** 2 + (0.5 * spread) ** 2))
    aligned = (ext + 90.0) % 180.0
    res["fine"] = {"angles_deg": [float(a) for a in fine_angles], "exposures": [float(e) for e in exposures], "fits": {f"{e:g}": v for e, v in fits.items()},
                   "exposure_used": float(best), "extinction_angle_stage_deg": float(ext), "extinction_angle_mod_180_deg": float(ext % 180.0),
                   "down_minus_up_deg": float(diff), "sigma_statistical_deg": stat, "spread_over_exposures_deg": spread, "sigma_deg": sigma,
                   "aligned_angle_deg": float(aligned), "aligned_angle_alternative_deg": float(aligned + 180.0)}
    print(f"  extinction at stage {ext % 180.0:.2f} deg (measured at {ext:.2f}); upward and downward differ by {diff:+.3f} deg; spread over the exposures {spread:.3f} deg")
    save()

    # ---- coarse check of the maximum
    print("\n=== Check of the maximum ===")
    chk_angles = [aligned + k for k in (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)]
    if chk_angles[0] < 0:
        chk_angles = [a + 180.0 for a in chk_angles]
    chk = scan(rig, chk_angles, [e0])
    pc = np.array(chk["by_exposure"][e0]["power"])
    q = np.polyfit(np.array(chk["angles"]), pc, 2)
    peak = float(-q[1] / (2 * q[0]))
    res["maximum_check"] = {"angles_deg": chk["angles"], "power": pc.tolist(), "exposure": e0, "parabola_maximum_deg": peak % 180.0,
                            "minus_aligned_deg": float((peak - aligned + 90.0) % 180.0 - 90.0), "saturated": bool(any(chk["by_exposure"][e0]["saturated"]))}
    print(f"  maximum of a parabola through 9 points: stage {peak % 180.0:.1f} deg ({res['maximum_check']['minus_aligned_deg']:+.1f} deg from extinction + 90 deg); "
          f"power at the maximum / lowest power of the fine scan: {pc.max() / max(min(fb['power_up']) * e0 / best, 1.0):.0f}")
    rig.move(aligned % 360.0)
    res["stage_left_at_deg"] = float(aligned % 360.0)
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    print(f"\nRESULT: the rotating polarizer is parallel to the static polarizer at stage {aligned:.2f} +- {sigma:.2f} deg (and at {aligned + 180.0:.2f} deg); "
          f"crossed at {ext % 180.0:.2f} deg. The stage is left at {aligned % 360.0:.2f} deg.")
    print(f"Saved to {out_dir}; warm-up frames discarded: {getattr(rig, 'n_warmup', 0)}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-index", type=int, default=0, help="index of the probe mask shown on the SLM")
    args = ap.parse_args()

    from run_tm_focus_optimization import PipelineConfig, TMFocusPipeline      # hardware libraries are loaded here

    pipeline = TMFocusPipeline(PipelineConfig())
    with h5py.File(pipeline.cfg.optimized_masks_h5_path, "r") as h5:
        mask = np.asarray(h5["mode_excitation_patterns"]["phase_masks_small_uint8"][args.mask_index], dtype=np.uint8)
    out_dir = Path(pipeline.cfg.output_dir) / f"static_polarizer_{datetime.now():%Y%m%d_%H%M}"
    stage = rpa.RotatingPolarizerStage()
    stage.connect()
    try:
        pipeline.setup()
        try:
            cam = pipeline.cameras[1]
            rig = rpf.GuardedRig(pipeline, stage, float(cam.ExposureTime.GetMin()))
            run(rig, mask, out_dir, float(cam.ExposureTime.GetMax()))
        finally:
            pipeline.teardown()
    finally:
        stage.close()


if __name__ == "__main__":
    main()
