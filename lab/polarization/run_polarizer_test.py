"""
Stage B (revised, 15 Nov 2026), step 3: polarizer test of a TM retrieved with camera weights.

The rotating polarizer (KDC101 stage) is in front of the transmission Fourier-plane camera.
Lab x = stage 0 deg, lab y = stage 90 deg. Model of the field at the polarizer:
E_pol = c D U E_model, D = diag(sqrt(1 + d), sqrt(1 - d)) in the stage axes (polarizer_arm_estimator.py).

    python run_polarizer_test.py --weighted-tm <checkpoint> --blind-tm <checkpoint> [--steps abcde]

Steps (letters of the plan):
  a  Displacement. At 0/45/90/135 deg: N_MAIN held-out probe masks, raw frames kept with a margin.
     Displacement = cross-correlation with the dataset frames of the same masks (no model).
     Checks: outline of the NA disk in the mean of the frames (no model, no speckle), and
     cross-correlation with the TM prediction. Every later frame is cropped from the raw camera
     frame around the displaced centre, so the NA disk is not cut. Other scan angles: N_SCAN masks.
  b  Freshness. (I_0 + I_90) against the dataset frames of the same masks; level factor.
     The run stops here if the mean correlation of the first 8 masks is below 0.9
     (unless --ignore-freshness).
  c  Axes without calibration: estimate_frame_with_diattenuation on the weighted TM.
  d  Calibration with foci: three two-foci patterns (PROBE_DESIGNS) at stage 0 and 90 deg,
     frame_from_focus_pairs. The x'/y' pattern is also test (a) of step e. The x'/y' pattern of the
     polarization-blind TM, test (c), is recorded in the same pass.
  e  Two foci with U of step d, test (b): recorded on the scan 0..165 deg in steps of 15 deg.
  f  Optional (plan 3f), only if the phase of step c is stable: two foci linearly polarized along stage 45 and
     135 deg, designed with the full frame of step c (U with its phase, d, theta0), in two variants:
     linear at the polarizer (behind the arm) and linear before the arm. Recorded on the scan 0..165 deg.
     In the same scan the x'/y' patterns of the weighted and of the blind TM are recorded again, which
     gives the azimuth and the degree of linear polarization of their foci, and 8 probe masks at 0 and
     90 deg give the freshness at that time. Run as: --steps cf --run-dir <folder of the first run>.
  q  Only if step b fails: a new dataset through the polarizer (--steps q --new-h5 <file> --level-factor <value>).

Focus power = sum over the 3x3 window at the brightest pixel near the target, dark level (median
outside the NA disk) removed, no speckle background removed. tau = P_0 / (P_0 + P_90).
"Before the arm": powers divided by (1 + d, 1 - d).

Everything is written to a new folder tm_focus_outputs/polarizer_test_<date>_<time>/.
Nothing in run_tm_focus_optimization.py, create_refs.py or the TM checkpoints is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import polarizer_arm_estimator as pae          # noqa: E402
import run_polarization_axes as rpa            # noqa: E402  (stage class, registration, two-foci optimizer, focus metrics)

MAIN_ANGLES = (0.0, 45.0, 90.0, 135.0)
SCAN_ANGLES = tuple(float(a) for a in range(0, 180, 15))
N_MAIN, N_SCAN, N_FIRST = 104, 16, 8           # masks at the main angles / at the other scan angles / "the 8 probe masks"
FIT_RADIUS = 50.1                              # px, region of the speckle fits (as in Stage A)
MARGIN_CAM = 110                               # camera pixels kept around the standard crop (25 px of the model grid)
MAX_SHIFT = 22.0                               # px of the model grid
TARGET_A, TARGET_B = (48, 56), (80, 72)        # (y, x) of the two foci, as on 12 Nov
SATURATION = 4095
FOCI_EXPOSURES = (24.0, 12.0, 6.0)             # one of these is used for all angles of a pattern: the highest without saturation


def disk(n: int, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:n, :n]
    return (yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2 <= radius ** 2


def held_out_indices(n_total: int, split: float = 0.95, seed: int = 42) -> np.ndarray:
    """Indices of the held-out probe masks of the pipeline (utils.load_real_mmf_data)."""
    from sklearn.model_selection import train_test_split
    _train, test = train_test_split(np.arange(n_total), test_size=1 - split, random_state=seed)
    return np.sort(test)


# =====================================================================================
# Hardware
# =====================================================================================
class Rig:
    """SLM and cameras through TMFocusPipeline, the rotation stage, and crops of the Fourier camera."""

    def __init__(self, pipeline, stage, min_exposure: float):
        self.pipeline, self.stage, self.min_exposure = pipeline, stage, float(min_exposure)
        self.n = int(pipeline.cfg.n_out)
        self.half = int(pipeline.n_pix_map["transmission_fourier"])          # half size of the standard crop, camera px
        self.yc = int(pipeline.y_centers["transmission_fourier"])
        self.xc = int(pipeline.x_centers["transmission_fourier"])
        self.cam_per_grid = 2.0 * self.half / self.n
        self.speckle_exposure = float(pipeline.cfg.transmission_fourier_et[0]) * float(pipeline.cfg.gain)

    def move(self, angle_deg: float) -> None:
        self.stage.move_to(float(angle_deg))

    def show(self, mask_small_u8: np.ndarray) -> None:
        self.pipeline.project_phase_mask_to_slm(self.pipeline.pack_phase_to_full_slm(mask_small_u8))

    def grab_fourier(self, exposure: float) -> np.ndarray:
        """Full sensor frame of the Fourier camera."""
        return np.squeeze(np.array(self.pipeline.capture_hdr_images(self.pipeline.cameras[1], [max(float(exposure), self.min_exposure)])))

    def grab_both(self) -> List[np.ndarray]:
        """Full sensor frames [image camera, Fourier camera] at the exposures of the TM acquisition."""
        return [np.squeeze(np.array(im)) for im in self.pipeline.capture_transmission_images()]

    def crop(self, raw: np.ndarray, offset_cam: Tuple[int, int] = (0, 0), half: Optional[int] = None) -> np.ndarray:
        half = self.half if half is None else int(half)
        y0, x0 = self.yc + int(offset_cam[0]) - half, self.xc + int(offset_cam[1]) - half
        if y0 < 0 or x0 < 0 or y0 + 2 * half > raw.shape[0] or x0 + 2 * half > raw.shape[1]:
            raise RuntimeError(f"Crop leaves the sensor: offset {offset_cam}, half size {half}, sensor {raw.shape}")
        return raw[y0:y0 + 2 * half, x0:x0 + 2 * half]

    def to_grid(self, cropped: np.ndarray, size: Optional[int] = None) -> np.ndarray:
        size = self.n if size is None else int(size)
        return cv2.resize(np.ascontiguousarray(cropped), (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# =====================================================================================
# Step a: displacement
# =====================================================================================
def outline_centre(mean_image: np.ndarray) -> Tuple[float, float]:
    """Centroid (y, x) of the region above half of the plateau level of a mean image (the NA disk)."""
    plateau = np.median(mean_image[mean_image > 0.2 * mean_image.max()])     # level of the bright region, whatever the dark margin
    yy, xx = np.nonzero(mean_image > 0.5 * plateau)
    return float(yy.mean()), float(xx.mean())


def measure_angle(rig: Rig, angle: float, masks: np.ndarray, ref_frames: np.ndarray, ref_envelope: np.ndarray,
                  model_frames: Optional[np.ndarray], exposure: float, with_outline: bool) -> Dict:
    """Record ``masks`` at one stage angle; return the displacement and the frames cropped around the displaced centre."""
    rig.move(angle)
    n, half, big_half = rig.n, rig.half, rig.half + MARGIN_CAM
    pad = int(round(MARGIN_CAM / rig.cam_per_grid))
    g = n + 2 * pad                                                        # size of the enlarged crop on the model grid
    big = np.empty((len(masks), 2 * big_half, 2 * big_half), dtype=np.uint16)
    saturated = 0
    for k in range(len(masks)):
        rig.show(masks[k])
        raw = rig.grab_fourier(exposure)
        saturated += int((raw >= SATURATION).sum())
        big[k] = rig.crop(raw, (0, 0), half=big_half)
    big_grid = np.stack([rig.to_grid(b, g) for b in big])                  # enlarged crop, model-grid scale (within 0.1 %)
    padded = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(pad, pad), (pad, pad)])
    d_data = np.array(rpa.estimate_shift(big_grid, padded(ref_frames), MAX_SHIFT)[:2])
    out = {"angle_deg": float(angle), "n_masks": int(len(masks)), "shift_from_dataset_frames_px": d_data.tolist(),
           "saturated_pixels_per_frame": saturated / len(masks), "mean_level": float(big_grid.mean())}
    if model_frames is not None:
        d_model = np.array(rpa.estimate_shift(big_grid, padded(model_frames), MAX_SHIFT)[:2])
        out["shift_from_tm_prediction_px"] = d_model.tolist()
        out["dataset_minus_tm_px"] = float(np.hypot(*(d_data - d_model)))
    if with_outline:
        mean_big = big.astype(np.float64).mean(axis=0)                     # camera pixels; speckle averaged out
        cy, cx = outline_centre(mean_big)
        ry, rx = outline_centre(ref_envelope)
        centre_cam, centre_grid = big_half - 0.5, (n - 1) / 2.0
        d_outline = np.array([(cy - centre_cam) / rig.cam_per_grid - (ry - centre_grid),
                              (cx - centre_cam) / rig.cam_per_grid - (rx - centre_grid)])
        d_env = np.array(rpa.estimate_shift(rig.to_grid(mean_big.astype(np.float32), g), padded(ref_envelope), MAX_SHIFT)[:2])
        out.update({"shift_from_disk_outline_px": d_outline.tolist(), "shift_from_mean_envelope_px": d_env.tolist(),
                    "dataset_minus_outline_px": float(np.hypot(*(d_data - d_outline)))})
    if np.abs(d_data).max() > MAX_SHIFT - 1.0:
        raise RuntimeError(f"Displacement {d_data} at {angle:g} deg reaches the search limit; re-centre the polarizer.")
    off = (int(round(d_data[0] * rig.cam_per_grid)), int(round(d_data[1] * rig.cam_per_grid)))
    y0, x0 = MARGIN_CAM + off[0], MARGIN_CAM + off[1]
    if min(y0, x0) < 0 or max(y0, x0) + 2 * half > 2 * big_half:
        raise RuntimeError(f"Displacement {d_data} px exceeds the recorded margin.")
    frames = np.stack([rig.to_grid(b[y0:y0 + 2 * half, x0:x0 + 2 * half]) for b in big])
    out["crop_offset_cam_px"] = [off[0], off[1]]
    out["residual_px"] = (d_data - np.array(off) / rig.cam_per_grid).tolist()
    return {"info": out, "frames": frames}


def step_a(rig: Rig, data: Dict, summary: Dict, out_dir: Path) -> Dict:
    print("\n=== 3a. Displacement of the image behind the polarizer ===")
    masks, ref, env, model = data["masks"], data["dataset_frames"], data["dataset_envelope"], data["tm_prediction"]
    frames, info, offsets = {}, {}, {}
    for ang in MAIN_ANGLES:
        r = measure_angle(rig, ang, masks[:N_MAIN], ref[:N_MAIN], env, model[:N_MAIN], rig.speckle_exposure, with_outline=True)
        frames[ang], info[ang] = r["frames"], r["info"]
    for ang in SCAN_ANGLES:
        if ang in MAIN_ANGLES:
            continue
        r = measure_angle(rig, ang, masks[:N_SCAN], ref[:N_SCAN], env, model[:N_SCAN], rig.speckle_exposure, with_outline=False)
        info[ang] = r["info"]
    rig.move(0.0)
    for ang in sorted(info):
        i = info[ang]
        offsets[ang] = tuple(i["crop_offset_cam_px"])
        line = (f"  {ang:5.0f} deg: ({i['shift_from_dataset_frames_px'][0]:+6.2f}, {i['shift_from_dataset_frames_px'][1]:+6.2f}) px from the dataset frames"
                f" | TM prediction differs by {i.get('dataset_minus_tm_px', float('nan')):.2f} px")
        if "shift_from_disk_outline_px" in i:
            line += (f" | disk outline ({i['shift_from_disk_outline_px'][0]:+6.2f}, {i['shift_from_disk_outline_px'][1]:+6.2f}), differs by "
                     f"{i['dataset_minus_outline_px']:.2f} px")
        print(line + f" | mean level {i['mean_level']:.1f}, saturated pixels per frame {i['saturated_pixels_per_frame']:.1f}")
    worst = max(info[a]["dataset_minus_outline_px"] for a in MAIN_ANGLES)
    d = np.array([info[a]["shift_from_dataset_frames_px"] for a in sorted(info)])
    a_rad = np.radians(sorted(info))
    coef = np.linalg.lstsq(np.stack([np.ones_like(a_rad), np.cos(a_rad), np.sin(a_rad)], axis=1), d, rcond=None)[0]
    radius = float(np.sqrt(0.5 * (coef[1] ** 2 + coef[2] ** 2).sum()))
    print(f"  largest difference between the two model-free estimates: {worst:.2f} px (limit 0.5 px); "
          f"circle of radius about {radius:.1f} px around ({coef[0][0]:+.1f}, {coef[0][1]:+.1f})")
    summary["3a"] = {"per_angle": {f"{a:g}": info[a] for a in sorted(info)}, "largest_dataset_minus_outline_px": worst,
                     "circle_radius_px": radius, "circle_centre_px": coef[0].tolist(), "agree_within_0p5_px": bool(worst <= 0.5),
                     "time": datetime.now().isoformat(timespec="seconds")}
    np.savez_compressed(out_dir / "speckle_frames_main_angles.npz", angles_deg=np.array(MAIN_ANGLES),
                        frames=np.stack([frames[a] for a in MAIN_ANGLES]).astype(np.float32),
                        mask_indices=data["mask_indices"][:N_MAIN], slm_masks_uint8=masks[:N_MAIN], exposure=rig.speckle_exposure)
    return {"frames": frames, "offsets": offsets}


# =====================================================================================
# Steps b and c: freshness, axes from speckle (no new hardware action)
# =====================================================================================
def step_b(frames: Dict[float, np.ndarray], data: Dict, summary: Dict) -> float:
    print("\n=== 3b. Freshness: I_0 + I_90 against the dataset frames of the same masks ===")
    n = frames[0.0].shape[-1]
    fit, support = disk(n, FIT_RADIUS), disk(n, data["support_radius"])
    ref = data["dataset_frames"][:N_MAIN]
    s = frames[0.0] + frames[90.0]
    s2 = frames[45.0] + frames[135.0]
    pcc_fit = np.array([rpa.pcc(s[k][fit], ref[k][fit]) for k in range(len(s))])
    pcc_sup = np.array([rpa.pcc(s[k][support], ref[k][support]) for k in range(len(s))])
    pcc2 = np.array([rpa.pcc(s2[k][fit], ref[k][fit]) for k in range(len(s))])
    level = np.array([s[k][support].mean() / ref[k][support].mean() for k in range(len(s))])
    level2 = np.array([s2[k][support].mean() / ref[k][support].mean() for k in range(len(s))])
    tm_pcc = np.array([rpa.pcc(data["tm_prediction"][k][fit], ref[k][fit]) for k in range(len(s))])
    mean_light = [float(frames[a][:, fit].mean()) for a in MAIN_ANGLES]
    s1, s2m, s0 = mean_light[0] - mean_light[2], mean_light[1] - mean_light[3], 0.5 * sum(mean_light)
    res = {"pcc_first_8_fit_region": pcc_fit[:N_FIRST].tolist(), "pcc_first_8_mean": float(pcc_fit[:N_FIRST].mean()),
           "pcc_all_mean": float(pcc_fit.mean()), "pcc_all_min": float(pcc_fit.min()), "pcc_all_mean_support": float(pcc_sup.mean()),
           "pcc_sum_45_135_mean": float(pcc2.mean()), "level_factor_first_8": float(level[:N_FIRST].mean()),
           "level_factor_all": float(level.mean()), "level_factor_45_135": float(level2.mean()),
           "tm_prediction_vs_dataset_pcc_mean": float(tm_pcc.mean()),
           "mean_light_counts_0_45_90_135": mean_light, "mean_light_share_along_stage_90": mean_light[2] / (mean_light[0] + mean_light[2]),
           "mean_light_dolp": float(np.hypot(s1, s2m) / s0), "mean_light_azimuth_deg": float(np.degrees(0.5 * np.arctan2(s2m, s1)) % 180.0),
           "minutes_since_dataset": data["minutes_since_dataset"](), "time": datetime.now().isoformat(timespec="seconds")}
    print(f"  first 8 masks, r <= {FIT_RADIUS:.0f} px: {[round(float(v), 3) for v in pcc_fit[:N_FIRST]]}, mean {res['pcc_first_8_mean']:.3f}")
    print(f"  all {len(s)} masks: mean {res['pcc_all_mean']:.3f} (lowest {res['pcc_all_min']:.3f}); whole NA disk: {res['pcc_all_mean_support']:.3f}; "
          f"I_45 + I_135: {res['pcc_sum_45_135_mean']:.3f}")
    print(f"  level factor (I_0 + I_90) / dataset frame: {res['level_factor_all']:.3f} (first 8: {res['level_factor_first_8']:.3f}; "
          f"I_45 + I_135: {res['level_factor_45_135']:.3f})")
    print(f"  for reference, weighted TM against the dataset frames of these masks: {res['tm_prediction_vs_dataset_pcc_mean']:.3f}")
    print(f"  mean light at 0/45/90/135 deg: {[round(v, 1) for v in mean_light]} counts; share along stage 90: "
          f"{res['mean_light_share_along_stage_90']:.3f}; degree of linear polarization {res['mean_light_dolp']:.3f}, azimuth {res['mean_light_azimuth_deg']:.1f} deg")
    print(f"  time since the dataset: {res['minutes_since_dataset']:.0f} min")
    summary["3b"] = res
    return res["pcc_first_8_mean"]


def frame_report(est: Dict) -> Dict:
    u, r = est["U"], est["R"]
    cross = float(min(abs(u[0, 0]) ** 2, abs(u[0, 1]) ** 2))
    return {"d": est["d"], "d_from_s2_amplitude": est["d_from_s2_amplitude"], "d_from_total_intensity": est["d_from_total_intensity"],
            "theta0_deg": est["theta0_deg"], "explained_fraction": est["explained_fraction"], "U_abs": np.abs(u).tolist(),
            "U_real": u.real.tolist(), "U_imag": u.imag.tolist(), "R": np.asarray(r).tolist(),
            "form": "identity" if abs(u[0, 0]) >= abs(u[0, 1]) else "swap", "cross_term_power": cross,
            "xprime_azimuth_deg": est["xprime_azimuth_deg"], "xprime_ellipticity_deg": est["xprime_ellipticity_deg"],
            # rotation of R about its first axis = phase between x' and y' (free in the retrieval)
            "phase_xy_deg": float(np.degrees(np.arctan2(r[2][1], r[1][1])))}


def step_c(frames: Dict[float, np.ndarray], data: Dict, summary: Dict) -> Dict:
    print("\n=== 3c. Axes without calibration (speckle, weighted TM) ===")
    n = frames[0.0].shape[-1]
    fit = disk(n, FIT_RADIUS)
    angles = list(MAIN_ANGLES)
    stack = np.stack([frames[a] for a in angles]).astype(np.float64)       # (4, N_MAIN, n, n)
    e = data["e_model"]                                                    # (N_MAIN, 2, n, n)
    subsets = {"first_8": slice(0, N_FIRST), "all": slice(0, N_MAIN), "half_1": slice(0, N_MAIN // 2), "half_2": slice(N_MAIN // 2, N_MAIN)}
    res = {}
    for name, sl in subsets.items():
        est = pae.estimate_frame_with_diattenuation(stack[:, sl], angles, e[sl], fit)
        rep = frame_report(est)
        rep["pcc_per_angle"] = [float(np.mean([rpa.pcc(pae.predict_polarizer_image(e[sl][k], est, a)[fit], stack[ia, sl][k][fit])
                                               for k in range(stack[:, sl].shape[1])])) for ia, a in enumerate(angles)]
        res[name] = rep
        if name == "first_8":
            res["_est_first_8"] = est
        if name == "all":
            res["_est_all"] = est
    for name in ("first_8", "all"):
        r = res[name]
        print(f"  {name}: |U| = [[{r['U_abs'][0][0]:.3f}, {r['U_abs'][0][1]:.3f}], [{r['U_abs'][1][0]:.3f}, {r['U_abs'][1][1]:.3f}]] ({r['form']}, "
              f"cross-term power {100 * r['cross_term_power']:.2f} %); x' azimuth {r['xprime_azimuth_deg']:.1f} deg, ellipticity {r['xprime_ellipticity_deg']:+.1f} deg; "
              f"theta0 {r['theta0_deg']:+.2f} deg")
        print(f"      d = {r['d']:+.3f} (S0 term), |d| = {r['d_from_s2_amplitude']:.3f} (I_45 - I_135), d = {r['d_from_total_intensity']:+.3f} (I_0 + I_90); "
              f"explained fraction {r['explained_fraction']:.3f}; correlation per angle {[round(v, 3) for v in r['pcc_per_angle']]}; "
              f"phase between x' and y' {r['phase_xy_deg']:+.1f} deg")
    dphi = (res["half_1"]["phase_xy_deg"] - res["half_2"]["phase_xy_deg"] + 180.0) % 360.0 - 180.0
    print(f"  phase from two halves of the masks: {res['half_1']['phase_xy_deg']:+.1f} and {res['half_2']['phase_xy_deg']:+.1f} deg (difference {dphi:+.1f} deg); "
          f"d: {res['half_1']['d']:+.3f} and {res['half_2']['d']:+.3f}")
    res["phase_difference_between_halves_deg"] = float(dphi)
    summary["3c"] = {k: v for k, v in res.items() if not k.startswith("_")}
    summary["3c"]["time"] = datetime.now().isoformat(timespec="seconds")
    return res


# =====================================================================================
# Steps d and e: foci
# =====================================================================================
def design_pattern(tm, u_design: np.ndarray, support: np.ndarray, seed: int = 0) -> np.ndarray:
    cfg = rpa.PolarizationConfig(plane="fourier", random_seed=seed)
    return rpa.optimize_two_foci(tm, u_design, TARGET_A, TARGET_B, support, cfg).cpu().numpy()


def record_patterns(rig: Rig, patterns: Dict[str, np.ndarray], angles: Sequence[float], offsets: Dict[float, Tuple[int, int]]) -> Dict:
    """Each pattern at every angle and every exposure of FOCI_EXPOSURES; per pattern, keep the highest exposure without saturation."""
    exposures = sorted({max(e, rig.min_exposure) for e in FOCI_EXPOSURES}, reverse=True)
    raw_frames = {name: {} for name in patterns}
    sat = {name: {e: False for e in exposures} for name in patterns}
    for ang in angles:
        rig.move(ang)
        for name, u8 in patterns.items():
            rig.show(u8)
            for e in exposures:
                raw = rig.grab_fourier(e)
                sat[name][e] |= bool(raw.max() >= SATURATION)
                raw_frames[name][(ang, e)] = rig.to_grid(rig.crop(raw, offsets[ang]))
    out = {}
    for name in patterns:
        ok = [e for e in exposures if not sat[name][e]]
        e_use = ok[0] if ok else exposures[-1]
        out[name] = {"exposure": float(e_use), "saturated": not ok,
                     "frames": {float(a): raw_frames[name][(a, e_use)] for a in angles}}
        if not ok:
            print(f"  WARNING: pattern {name} saturates at every exposure; the metrics are biased.")
    return out


def foci_metrics(rec: Dict, support: np.ndarray, d_arm: Optional[float] = None) -> Dict:
    m = rpa.focus_metrics(rec["frames"], TARGET_A, TARGET_B, support)
    i0, i90 = m["angles_deg"].index(0.0), m["angles_deg"].index(90.0)
    pa, pb = m["A"]["window_power"], m["B"]["window_power"]
    m["powers_A_x_y_B_x_y"] = [pa[i0], pa[i90], pb[i0], pb[i90]]
    m["pbr_A_at_0_90"] = [m["A"]["pbr_behind_polarizer"][i0], m["A"]["pbr_behind_polarizer"][i90]]
    m["pbr_B_at_0_90"] = [m["B"]["pbr_behind_polarizer"][i0], m["B"]["pbr_behind_polarizer"][i90]]
    m["exposure"], m["saturated"] = rec["exposure"], rec["saturated"]
    if d_arm is not None:
        m["before_arm"] = before_arm(m["powers_A_x_y_B_x_y"], d_arm)
    return m


def before_arm(p: Sequence[float], d: float) -> Dict:
    est = {"weights_xy": [1.0 + d, 1.0 - d]}
    ta, tb = pae.pass_fraction_before_arm(p[0], p[1], est), pae.pass_fraction_before_arm(p[2], p[3], est)
    return {"d_used": float(d), "tau_A": float(ta), "tau_B": float(tb), "separation": float(ta - tb)}


def print_foci(name: str, m: Dict) -> None:
    line = (f"  {name}: A passes {m['transmitted_fraction_A_at_x']:.3f} at stage 0, B passes {m['transmitted_fraction_B_at_x']:.3f}; "
            f"separation {m['separation']:+.3f} raw")
    if "before_arm" in m:
        b = m["before_arm"]
        line += f", {b['separation']:+.3f} before the arm (d = {b['d_used']:+.3f}: A {b['tau_A']:.3f}, B {b['tau_B']:.3f})"
    print(line)
    print(f"      leak B/A at stage 0: {m['leak_B_over_A_at_x']:.3f}; leak A/B at stage 90: {m['leak_A_over_B_at_y']:.3f}; "
          f"PBR A {m['pbr_A_at_0_90'][0]:.0f} / {m['pbr_A_at_0_90'][1]:.0f}, B {m['pbr_B_at_0_90'][0]:.0f} / {m['pbr_B_at_0_90'][1]:.0f} (stage 0 / 90); "
          f"peaks A {m['A']['peak_yx']}, B {m['B']['peak_yx']}; exposure {m['exposure']:g}{' SATURATED' if m['saturated'] else ''}")


def save_pattern(rig: Rig, out_dir: Path, tag: str, phase_rad: np.ndarray) -> np.ndarray:
    u8 = rig.pipeline.phase_to_uint8(phase_rad)
    np.save(out_dir / f"two_foci_{tag}_phase_rad.npy", phase_rad)
    np.save(out_dir / f"two_foci_{tag}_phase_uint8.npy", u8)
    np.save(out_dir / f"two_foci_{tag}_full_slm_mask_uint8.npy", rig.pipeline.pack_phase_to_full_slm(u8))
    if hasattr(rig.pipeline, "save_phase_hsv_png"):
        rig.pipeline.save_phase_hsv_png(phase_rad, out_dir / f"two_foci_{tag}_phase_hsv.png")
    return u8


def save_frames(out_dir: Path, tag: str, rec: Dict) -> None:
    angles = sorted(rec["frames"])
    np.savez_compressed(out_dir / f"two_foci_{tag}_frames.npz", angles_deg=np.array(angles),
                        frames=np.stack([rec["frames"][a] for a in angles]), exposure=rec["exposure"])
    rpa._save_png16_common({a: rec["frames"][a] for a in angles if a in (0.0, 90.0)}, out_dir, f"im_two_foci_{tag}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = [a for a in angles if a in (0.0, 90.0)] if len(angles) > 4 else angles
        fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.6), squeeze=False)
        vmax = max(float(rec["frames"][a].max()) for a in show)
        for ax, a in zip(axes[0], show):
            ax.imshow(rec["frames"][a], cmap="inferno", vmin=0, vmax=vmax)
            for tgt, colour in ((TARGET_A, "cyan"), (TARGET_B, "lime")):
                ax.plot(tgt[1], tgt[0], marker="o", ms=14, mfc="none", mec=colour, mew=0.8)
            ax.set_title(f"{tag}: polarizer at stage {a:g} deg", fontsize=8)
            ax.set_axis_off()
        fig.suptitle("cyan: focus A   green: focus B   (one colour scale)", fontsize=8)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"two_foci_{tag}_overview.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  (overview figure not saved: {exc})")


def step_d(rig: Rig, tm_w, tm_blind, offsets: Dict, data: Dict, summary: Dict, out_dir: Path) -> Dict:
    print("\n=== 3d. Calibration with foci (weighted TM), and the x'/y' pattern of the blind TM ===")
    support = disk(rig.n, data["support_radius"])
    patterns, tags = {}, {"x'/y'": "probe_xy_weighted_tm", "+45'/-45'": "probe_pm45_weighted_tm", "R'/L'": "probe_RL_weighted_tm"}
    for name, design in pae.PROBE_DESIGNS.items():
        print(f"  designing the {name} pattern")
        patterns[tags[name]] = save_pattern(rig, out_dir, tags[name], design_pattern(tm_w, design, support))
    if tm_blind is not None:
        print("  designing the x'/y' pattern of the polarization-blind TM")
        patterns["xy_blind_tm"] = save_pattern(rig, out_dir, "xy_blind_tm", design_pattern(tm_blind, np.eye(2, dtype=complex), support))
    rec = record_patterns(rig, patterns, (0.0, 90.0), offsets)
    rig.move(0.0)
    metrics = {tag: foci_metrics(rec[tag], support) for tag in patterns}
    for tag in patterns:
        save_frames(out_dir, tag, rec[tag])
    powers = {name: metrics[tags[name]]["powers_A_x_y_B_x_y"] for name in pae.PROBE_DESIGNS}
    est = pae.frame_from_focus_pairs(powers)
    b = data["b"]
    r1 = np.array(est["R1"])
    ang_plus = float(np.degrees(np.arccos(np.clip(r1[0], -1, 1))))
    # A pair whose foci are pure lab x / lab y (u = +-1) carries no information on d; its d_k is meaningless.
    # Compare the d_k only among the pairs that constrain d (1 - u^2 > 0.2), and give their weighted mean as well.
    p_k, d_k = np.array(est["p"]), np.array(est["d_per_basis"])
    u_k = 0.5 * ((p_k[:, 0] - d_k) / (1 - d_k * p_k[:, 0]) - (p_k[:, 1] - d_k) / (1 - d_k * p_k[:, 1]))
    w_k = 1.0 - u_k ** 2
    use = w_k > 0.2
    d_constraining = float((w_k[use] * d_k[use]).sum() / w_k[use].sum()) if use.any() else float(est["d"])
    checks = {"R1_norm_within_0p9_1p1": bool(0.9 <= est["R1_norm"] <= 1.1),
              "pair_imbalance_below_0p15": bool(np.max(np.abs(est["pair_imbalance"])) < 0.15),
              "d_per_basis_agree_within_0p15": bool(use.sum() < 2 or np.ptp(d_k[use]) <= 0.15)}
    res = {"d_from_constraining_pairs": d_constraining, "pairs_that_constrain_d": [n for n, f in zip(pae.PROBE_DESIGNS, use) if f],
           "d": est["d"], "d_per_basis": est["d_per_basis"], "R1": est["R1"], "R1_norm": est["R1_norm"],
           "pair_imbalance": est["pair_imbalance"], "p": est["p"], "U_real": est["U"].real.tolist(), "U_imag": est["U"].imag.tolist(),
           "angle_R1_to_plus_e1_deg": ang_plus, "angle_R1_to_minus_e1_deg": 180.0 - ang_plus,
           "b_of_weighted_tm": b, "d_minus_plus_b": est["d"] - b, "d_minus_minus_b": est["d"] + b, "checks": checks,
           "powers": powers, "patterns": metrics, "minutes_since_dataset": data["minutes_since_dataset"](),
           "time": datetime.now().isoformat(timespec="seconds")}
    for tag in patterns:
        metrics[tag]["before_arm"] = before_arm(metrics[tag]["powers_A_x_y_B_x_y"], est["d"])
        print_foci(tag, metrics[tag])
    print(f"  frame_from_focus_pairs: d = {est['d']:+.3f} (per basis {[round(v, 3) for v in est['d_per_basis']]}); "
          f"from the pairs that constrain d ({', '.join(res['pairs_that_constrain_d'])}): {d_constraining:+.3f}; learned b = {b:+.3f}")
    print(f"  R1 = ({r1[0]:+.3f}, {r1[1]:+.3f}, {r1[2]:+.3f}), |R1| before normalization {est['R1_norm']:.3f}; "
          f"angle to (+1, 0, 0): {ang_plus:.1f} deg, to (-1, 0, 0): {180 - ang_plus:.1f} deg")
    print(f"  pair imbalance: {[round(v, 3) for v in est['pair_imbalance']]}; checks: {checks}")
    summary["3d"] = res
    np.save(out_dir / "jones_matrix_U_from_foci.npy", est["U"])
    return {"est": est, "metrics": metrics, "checks_ok": all(checks.values())}


def step_e(rig: Rig, tm_w, u_foci: np.ndarray, d_foci: float, offsets: Dict, data: Dict, summary: Dict, out_dir: Path) -> None:
    print("\n=== 3e (b). Two foci designed with U from the foci calibration; scan 0..165 deg ===")
    support = disk(rig.n, data["support_radius"])
    tag = "lab_axes_U_from_foci"
    patterns = {tag: save_pattern(rig, out_dir, tag, design_pattern(tm_w, u_foci, support))}
    rec = record_patterns(rig, patterns, SCAN_ANGLES, offsets)[tag]
    rig.move(0.0)
    m = foci_metrics(rec, support, d_foci)
    save_frames(out_dir, tag, rec)
    print_foci(tag, m)
    for name in ("A", "B"):
        print(f"      scan, focus {name}: azimuth {m[name]['azimuth_deg']:.1f} deg, degree of linear polarization {m[name]['degree_of_linear_polarization']:.3f}; "
              f"power per angle {[round(v) for v in m[name]['window_power']]}")
    m["minutes_since_dataset"] = data["minutes_since_dataset"]()
    m["time"] = datetime.now().isoformat(timespec="seconds")
    summary["3e_b"] = m


# =====================================================================================
# Step f (optional): foci at +-45 deg with the phase from the speckle estimate; scans of the x'/y' patterns
# =====================================================================================
def projection_design(est: Dict, stage_angles: Sequence[float], with_arm: bool) -> np.ndarray:
    """
    2 x 2 matrix whose row k gives the field component that a polarizer at stage angle stage_angles[k]
    passes: P(angle) . D . U (with_arm) or P(angle) . U (without the arm). est: frame of the speckle
    estimate (U, d, theta0); its lab x axis lies at stage angle theta0.
    """
    u, d, th = np.asarray(est["U"]), float(est["d"]), float(est["theta0_deg"])
    arm = np.diag([np.sqrt(1.0 + d), np.sqrt(1.0 - d)]) if with_arm else np.eye(2)
    rows = [np.array([np.cos(np.radians(a - th)), np.sin(np.radians(a - th))]) for a in stage_angles]
    return np.stack(rows) @ arm @ u


def pm45_numbers(m: Dict) -> Dict:
    i45, i135 = m["angles_deg"].index(45.0), m["angles_deg"].index(135.0)
    pa, pb = m["A"]["window_power"], m["B"]["window_power"]
    ta, tb = pa[i45] / (pa[i45] + pa[i135]), pb[i45] / (pb[i45] + pb[i135])
    return {"tau45_A": float(ta), "tau45_B": float(tb), "separation_45_135": float(ta - tb),
            "leak_B_over_A_at_45": float(pb[i45] / pa[i45]), "leak_A_over_B_at_135": float(pa[i135] / pb[i135]),
            "pbr_A_at_45_135": [m["A"]["pbr_behind_polarizer"][i45], m["A"]["pbr_behind_polarizer"][i135]],
            "pbr_B_at_45_135": [m["B"]["pbr_behind_polarizer"][i45], m["B"]["pbr_behind_polarizer"][i135]]}


def predicted_scan(tm, u8: np.ndarray, est: Dict, angles: Sequence[float]) -> Dict:
    """Model prediction for a displayed pattern behind the polarizer, with the frame of the speckle estimate."""
    n = u8.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    with torch.no_grad():
        ef = rpa.model_plane_field(tm, rpa._phase_from_uint8(tm, u8[None]), "fourier")[0].cpu().numpy()
    out = {}
    for name, tgt in (("A", TARGET_A), ("B", TARGET_B)):
        win = (np.abs(yy - tgt[0]) <= 1) & (np.abs(xx - tgt[1]) <= 1)
        pw = np.array([pae.predict_polarizer_image(ef, est, a)[win].sum() for a in angles])
        a_rad = np.radians(np.array(angles))
        c = np.linalg.lstsq(0.5 * np.stack([np.ones_like(a_rad), np.cos(2 * a_rad), np.sin(2 * a_rad)], axis=1), pw, rcond=None)[0]
        out[name] = {"azimuth_deg": float(np.degrees(0.5 * np.arctan2(c[2], c[1])) % 180.0),
                     "degree_of_linear_polarization": float(np.hypot(c[1], c[2]) / c[0]),
                     "tau_0_90": float(pw[list(angles).index(0.0)] / (pw[list(angles).index(0.0)] + pw[list(angles).index(90.0)])),
                     "tau_45_135": float(pw[list(angles).index(45.0)] / (pw[list(angles).index(45.0)] + pw[list(angles).index(135.0)]))}
    return out


def step_f(rig: Rig, tm_w, tm_blind, est_c: Dict, offsets: Dict, data: Dict, summary: Dict, out_dir: Path) -> None:
    print("\n=== 3f. Foci at stage 45 / 135 deg with the phase of step c; scans of the x'/y' patterns ===")
    support = disk(rig.n, data["support_radius"])
    est = {"U": est_c["U"], "d": est_c["d"], "theta0_deg": est_c["theta0_deg"]}
    patterns, model_of = {}, {}
    for tag, with_arm in (("pm45_at_polarizer", True), ("pm45_before_arm", False)):
        print(f"  designing {tag}")
        patterns[tag] = save_pattern(rig, out_dir, tag, design_pattern(tm_w, projection_design(est, (45.0, 135.0), with_arm), support))
        model_of[tag] = tm_w
    for tag, src, tm in (("scan_probe_xy_weighted_tm", "probe_xy_weighted_tm", tm_w), ("scan_xy_blind_tm", "xy_blind_tm", tm_blind)):
        f = out_dir / f"two_foci_{src}_phase_uint8.npy"
        if f.exists() and tm is not None:
            patterns[tag], model_of[tag] = np.load(f), tm
    # one pass over the scan angles; at 0 and 90 deg also 8 probe masks for the freshness
    exposures = sorted({max(e, rig.min_exposure) for e in FOCI_EXPOSURES}, reverse=True)
    raw_frames = {name: {} for name in patterns}
    sat = {name: {e: False for e in exposures} for name in patterns}
    fresh = {}
    for ang in SCAN_ANGLES:
        rig.move(ang)
        for name, u8 in patterns.items():
            rig.show(u8)
            for e in exposures:
                raw = rig.grab_fourier(e)
                sat[name][e] |= bool(raw.max() >= SATURATION)
                raw_frames[name][(ang, e)] = rig.to_grid(rig.crop(raw, offsets[ang]))
        if ang in (0.0, 90.0):
            fr = []
            for k in range(N_FIRST):
                rig.show(data["masks"][k])
                fr.append(rig.to_grid(rig.crop(rig.grab_fourier(rig.speckle_exposure), offsets[ang])))
            fresh[ang] = np.stack(fr)
    rig.move(0.0)
    fit = disk(rig.n, FIT_RADIUS)
    ssum = fresh[0.0] + fresh[90.0]
    pcc = [rpa.pcc(ssum[k][fit], data["dataset_frames"][k][fit]) for k in range(N_FIRST)]
    res = {"freshness_first_8": pcc, "freshness_mean": float(np.mean(pcc)), "minutes_since_dataset": data["minutes_since_dataset"](),
           "frame_used": {"d": est["d"], "theta0_deg": est["theta0_deg"], "phase_xy_deg": float(np.degrees(np.arctan2(est_c["R"][2][1], est_c["R"][1][1])))},
           "time": datetime.now().isoformat(timespec="seconds"), "patterns": {}}
    print(f"  freshness now (8 masks, I_0 + I_90 against the dataset): mean {res['freshness_mean']:.3f}; {res['minutes_since_dataset']:.0f} min since the dataset")
    for name in patterns:
        ok = [e for e in exposures if not sat[name][e]]
        e_use = ok[0] if ok else exposures[-1]
        rec = {"exposure": float(e_use), "saturated": not ok, "frames": {float(a): raw_frames[name][(a, e_use)] for a in SCAN_ANGLES}}
        m = foci_metrics(rec, support)
        m.update(pm45_numbers(m))
        est_pred = est if model_of[name] is tm_w else None
        if est_pred is not None:
            m["predicted_with_speckle_frame"] = predicted_scan(tm_w, patterns[name], est_pred, SCAN_ANGLES)
        save_frames(out_dir, name, rec)
        res["patterns"][name] = m
        print(f"  {name}: exposure {m['exposure']:g}{' SATURATED' if m['saturated'] else ''}; peaks A {m['A']['peak_yx']}, B {m['B']['peak_yx']}")
        for foc in ("A", "B"):
            line = (f"      focus {foc}: azimuth {m[foc]['azimuth_deg']:.1f} deg, degree of linear polarization {m[foc]['degree_of_linear_polarization']:.3f}; "
                    f"passes {m['tau45_' + foc]:.3f} at stage 45 (of 45 + 135), {m['transmitted_fraction_' + foc + '_at_x']:.3f} at stage 0 (of 0 + 90)")
            if "predicted_with_speckle_frame" in m:
                q = m["predicted_with_speckle_frame"][foc]
                line += f" | predicted: azimuth {q['azimuth_deg']:.1f} deg, DoLP {q['degree_of_linear_polarization']:.3f}, {q['tau_45_135']:.3f}, {q['tau_0_90']:.3f}"
            print(line)
        print(f"      separation 45/135: {m['separation_45_135']:+.3f}; separation 0/90: {m['separation']:+.3f}; "
              f"PBR at 45 / 135: A {m['pbr_A_at_45_135'][0]:.0f} / {m['pbr_A_at_45_135'][1]:.0f}, B {m['pbr_B_at_45_135'][0]:.0f} / {m['pbr_B_at_45_135'][1]:.0f}")
    summary["3f"] = res


# =====================================================================================
# Step q (only if step b fails): new dataset through the polarizer
# =====================================================================================
def acquire_dataset_through_polarizer(rig: Rig, data: Dict, masks_h5: Path, out_h5: Path, level_factor: float, summary: Dict) -> None:
    """
    Plan step 3b: all probe masks with the stage at 0 deg, then at 90 deg, both cameras each time.
    Fourier frame = sum of the two passes (each cropped around its displaced centre) / level_factor;
    image frame = mean of the two passes. Written to ``out_h5`` in the format of the pipeline.
    """
    from tqdm import tqdm
    print(f"\n=== 3b. New dataset through the polarizer -> {out_h5} ===")
    if out_h5.exists():
        raise FileExistsError(f"{out_h5} exists; choose a new name.")
    with h5py.File(masks_h5, "r") as h5:
        masks = np.asarray(h5["mode_excitation_patterns"]["phase_masks_small_uint8"][:], dtype=np.uint8)
    n_total, n = len(masks), rig.n
    four = np.zeros((2, n_total, n, n), dtype=np.float32)
    im = np.zeros((2, n_total, n, n), dtype=np.float32)
    pm = np.zeros((n_total, n, n), dtype=np.uint8)
    offsets, shifts, sat_frac, t_pass = {}, {}, [], []
    for ip, ang in enumerate((0.0, 90.0)):
        r = measure_angle(rig, ang, data["masks"][:N_SCAN], data["dataset_frames"][:N_SCAN], data["dataset_envelope"],
                          data["tm_prediction"][:N_SCAN], rig.speckle_exposure, with_outline=False)
        offsets[ang], shifts[ang] = tuple(r["info"]["crop_offset_cam_px"]), r["info"]["shift_from_dataset_frames_px"]
        print(f"  stage {ang:g} deg: displacement ({shifts[ang][0]:+.2f}, {shifts[ang][1]:+.2f}) px")
        t_pass.append(datetime.now().isoformat(timespec="seconds"))
        saturated = 0
        for i in tqdm(range(n_total), desc=f"stage {ang:g} deg"):
            full = rig.pipeline.pack_phase_to_full_slm(masks[i])
            rig.pipeline.project_phase_mask_to_slm(full)
            image_raw, fourier_raw = rig.grab_both()
            saturated += int((fourier_raw >= SATURATION).sum())
            four[ip, i] = rig.to_grid(rig.crop(fourier_raw, offsets[ang]))
            im[ip, i] = rig.pipeline.crop_and_resize_image(image_raw, "transmission_image")
            if ip == 0:
                pm[i] = np.round(rig.pipeline.crop_and_resize_image(full, "PhaseMask")).astype(np.uint8)
        sat_frac.append(saturated / (n_total * (2 * rig.half) ** 2))
    rig.move(0.0)
    fourier = (four[0] + four[1]) / float(level_factor)
    image = 0.5 * (im[0] + im[1])
    drift = np.array([rpa.pcc(im[0][i], im[1][i]) for i in range(0, n_total, 7)])
    u16 = lambda a: np.clip(np.round(a), 0, 65535).astype(np.uint16)
    with h5py.File(out_h5, "w") as h5:
        g = h5.create_group("experiment_0")
        g.create_dataset("transmission_image", data=u16(image), chunks=True)
        g.create_dataset("transmission_fourier", data=u16(fourier), chunks=True)
        g.create_dataset("PhaseMask", data=pm, chunks=True)
        p = h5.create_group("polarizer_passes")            # the single passes, for later analysis
        p.create_dataset("fourier_stage_0", data=u16(four[0]), chunks=True, compression="gzip", compression_opts=1)
        p.create_dataset("fourier_stage_90", data=u16(four[1]), chunks=True, compression="gzip", compression_opts=1)
        p.create_dataset("image_pass_stage_0", data=u16(im[0]), chunks=True, compression="gzip", compression_opts=1)
        p.create_dataset("image_pass_stage_90", data=u16(im[1]), chunks=True, compression="gzip", compression_opts=1)
        p.attrs["level_factor"] = float(level_factor)
    res = {"file": str(out_h5), "n_masks": int(n_total), "level_factor": float(level_factor), "pass_start_times": t_pass,
           "displacement_px": {f"{a:g}": shifts[a] for a in shifts}, "saturated_pixel_fraction_per_pass": sat_frac,
           "image_pass_correlation_mean": float(drift.mean()), "image_pass_correlation_min": float(drift.min()),
           "finished": datetime.now().isoformat(timespec="seconds")}
    print(f"  image plane, pass at stage 0 against pass at stage 90 (drift): mean {res['image_pass_correlation_mean']:.3f}, lowest {res['image_pass_correlation_min']:.3f}")
    print(f"  saturated pixels per pass: {[f'{100 * v:.3f} %' for v in sat_frac]}")
    summary["3b_new_dataset"] = res


# =====================================================================================
# Data that do not need the hardware
# =====================================================================================
def load_tm(path: Path, device: str):
    tm = torch.load(path, map_location=device, weights_only=False)
    tm.device = device
    tm.eval()
    for p in tm.parameters():
        p.requires_grad = False
    return tm


def prepare_data(tm_w, h5_path: Path, masks_h5: Path, support_radius: float) -> Dict:
    with h5py.File(h5_path, "r") as h5:
        g = h5["experiment_0"]
        n_total = int(g["transmission_fourier"].shape[0])
        idx = held_out_indices(n_total)[:N_MAIN]
        dataset_frames = np.asarray(g["transmission_fourier"][idx], dtype=np.float32)          # Fourier plane: no flip
        envelope = np.zeros(dataset_frames.shape[1:], dtype=np.float64)
        for i in range(0, n_total, 500):
            envelope += np.asarray(g["transmission_fourier"][i:i + 500], dtype=np.float64).sum(axis=0)
        envelope /= n_total
    with h5py.File(masks_h5, "r") as h5:
        masks = np.asarray(h5["mode_excitation_patterns"]["phase_masks_small_uint8"][idx], dtype=np.uint8)
    with torch.no_grad():
        phase = rpa._phase_from_uint8(tm_w, masks)
        e = torch.cat([rpa.model_plane_field(tm_w, phase[i:i + 16], "fourier") for i in range(0, len(masks), 16)]).cpu().numpy()
        pred = torch.cat([tm_w.forward_model_slm(phase[i:i + 16])[:, 1] for i in range(0, len(masks), 16)]).cpu().numpy()
    t_dataset = h5_path.stat().st_mtime
    b = float(tm_w.fourier_weight_b()) if hasattr(tm_w, "fourier_weight_b") else 0.0
    return {"mask_indices": idx, "masks": masks, "dataset_frames": dataset_frames, "dataset_envelope": envelope, "e_model": e,
            "tm_prediction": pred, "support_radius": float(support_radius), "b": b,
            "minutes_since_dataset": lambda: (time.time() - t_dataset) / 60.0,
            "dataset_time": datetime.fromtimestamp(t_dataset).isoformat(timespec="seconds")}


def run_test(rig: Rig, tm_w, tm_blind, data: Dict, steps: str, out_dir: Path, ignore_freshness: bool = False) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict = {"started": datetime.now().isoformat(timespec="seconds"), "dataset_time": data["dataset_time"],
                     "b_of_weighted_tm": data["b"], "targets_yx": {"A": list(TARGET_A), "B": list(TARGET_B)},
                     "mask_indices": [int(i) for i in data["mask_indices"]], "fit_radius_px": FIT_RADIUS}
    previous = out_dir / "polarizer_test_summary.json"
    if previous.exists() and "a" not in steps:            # continuing in an existing folder: keep what is there
        with open(previous, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary.setdefault("continued", []).append({"steps": steps, "time": datetime.now().isoformat(timespec="seconds")})

    def save() -> None:
        with open(out_dir / "polarizer_test_summary.json", "w", encoding="utf-8") as f:
            json.dump(rpa._to_jsonable(summary), f, indent=1)

    if "q" in steps:                                   # only the new dataset; the test itself is then run again on the new TM
        acquire_dataset_through_polarizer(rig, data, data["masks_h5"], data["new_h5"], data["level_factor"], summary)
        save()
        return summary
    state_path = out_dir / "state_step_a.npz"
    if "a" in steps:
        a = step_a(rig, data, summary, out_dir)
        np.savez_compressed(state_path, angles=np.array(sorted(a["offsets"])), offsets=np.array([a["offsets"][k] for k in sorted(a["offsets"])]))
        save()
    else:
        st = np.load(state_path)
        a = {"offsets": {float(k): tuple(int(v) for v in o) for k, o in zip(st["angles"], st["offsets"])},
             "frames": dict(zip(MAIN_ANGLES, np.load(out_dir / "speckle_frames_main_angles.npz")["frames"]))}
    if "b" in steps:
        fresh = step_b(a["frames"], data, summary)
        save()
        if fresh < 0.9 and not ignore_freshness:
            print("\nFRESHNESS BELOW 0.9: stopping before the calibration. A new dataset through the polarizer is needed (plan, step 3b).")
            summary["stopped"] = "freshness below 0.9"
            save()
            return summary
    c = None
    if "c" in steps:
        c = step_c(a["frames"], data, summary)
        save()
    if "d" in steps:
        d = step_d(rig, tm_w, tm_blind, a["offsets"], data, summary, out_dir)
        save()
        if "e" in steps:
            if not d["checks_ok"]:
                print("\nThe checks of the foci calibration failed: test (b) is recorded all the same, but read it with care.")
            step_e(rig, tm_w, d["est"]["U"], d["est"]["d"], a["offsets"], data, summary, out_dir)
            save()
    if "f" in steps:
        if c is None:
            raise SystemExit("Step f needs step c in the same call (--steps cf).")
        step_f(rig, tm_w, tm_blind, c["_est_all"], a["offsets"], data, summary, out_dir)
        save()
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    print(f"\nSaved everything to {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weighted-tm", required=True, help="TM retrieved with camera weights (TM_retrieve_camweights)")
    ap.add_argument("--blind-tm", default="", help="polarization-blind TM for the baseline (c); empty = skip")
    ap.add_argument("--h5", default=str(TM_LIB / "measurement_focus.h5"), help="dataset the TMs were retrieved from")
    ap.add_argument("--steps", default="abcde")
    ap.add_argument("--run-dir", default="", help="continue in an existing output folder (for steps without 'a')")
    ap.add_argument("--ignore-freshness", action="store_true")
    ap.add_argument("--new-h5", default="", help="step q: file name of the new dataset recorded through the polarizer")
    ap.add_argument("--level-factor", type=float, default=0.0, help="step q: (I_0 + I_90) / frame without polarizer, from step b")
    args = ap.parse_args()

    from run_tm_focus_optimization import PipelineConfig, TMFocusPipeline      # hardware libraries are loaded here

    pipeline = TMFocusPipeline(PipelineConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tm_w = load_tm(Path(args.weighted_tm), device)
    tm_blind = load_tm(Path(args.blind_tm), device) if args.blind_tm else None
    data = prepare_data(tm_w, Path(args.h5), Path(pipeline.cfg.optimized_masks_h5_path), pipeline.support_radius_px("fourier"))
    data.update({"masks_h5": Path(pipeline.cfg.optimized_masks_h5_path), "new_h5": Path(args.new_h5) if args.new_h5 else None,
                 "level_factor": args.level_factor})
    if "q" in args.steps and (not args.new_h5 or args.level_factor <= 0):
        raise SystemExit("Step q needs --new-h5 and --level-factor.")
    print(f"Weighted TM: {args.weighted_tm} (b = {data['b']:+.4f}); dataset of {data['dataset_time']}, {data['minutes_since_dataset']():.0f} min old")
    out_dir = Path(args.run_dir) if args.run_dir else Path(pipeline.cfg.output_dir) / f"polarizer_test_{datetime.now():%Y%m%d_%H%M}"

    stage = rpa.RotatingPolarizerStage()
    stage.connect()                              # before the cameras and the SLM, so a missing stage fails early
    try:
        pipeline.setup()
        try:
            from create_refs import get_camera_exposure_minimum
            rig = Rig(pipeline, stage, float(get_camera_exposure_minimum(pipeline.cameras[1])))
            print(f"Fourier camera {pipeline.cameras[1].GetDeviceInfo().GetSerialNumber()}: minimum exposure {rig.min_exposure:g}, "
                  f"speckle exposure {rig.speckle_exposure:g}")
            run_test(rig, tm_w, tm_blind, data, args.steps, out_dir, ignore_freshness=args.ignore_freshness)
        finally:
            pipeline.teardown()
    finally:
        stage.close()


if __name__ == "__main__":
    main()
