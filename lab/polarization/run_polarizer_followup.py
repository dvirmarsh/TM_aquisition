"""
Follow-up of the polarizer test of 15 Nov 2026 (run_polarizer_test.py). Version 2 (15 Nov, afternoon).

    python run_polarizer_followup.py --weighted-tm <checkpoint> --first-run <folder of the first run> [--x-angle 6]

Guard against a one-frame lag between the SLM and the camera: the rig takes one warm-up frame, which is
discarded, after every change of the stage angle, of the SLM pattern or of the exposure (FrameGuard).
The freshness step also reports how the frames correlate with the dataset frame of the PREVIOUS mask.

Steps
 1. Freshness: 16 probe masks at stage 0 and 90 deg; (I_0 + I_90) of the first 8 against the dataset frames.
    A value below 0.9 is measured once more; the run stops if it is below 0.9 twice.
 2. Displacement at more stage angles; one circle d(angle) = c + p cos(angle) + q sin(angle) through all
    measured angles gives the crop of every later frame.
 3. Fine scans of pattern (a) (x'/y' of the weighted TM, the SLM pattern of the first run), 2 deg steps, three
    exposures, each range upward and then downward:
      low range   -10 .. 20 deg  (stage 350 .. 358, then 0 .. 20)
      high range   80 .. 110 deg
      low range again at stage angle + 180 deg (170 .. 200): the same analyzer, used to check that equivalence.
    The extinction angle of a focus is the minimum of P(angle) = c0 + c1 cos(2 angle) + c2 sin(2 angle) fitted to
    the range in which it goes through extinction, at the longest exposure at which that focus does not saturate.
    Upward and downward must agree within 0.5 deg, otherwise the run stops.
 4. Full scan of pattern (a), 0 .. 175 deg in steps of 5 deg, one exposure, with a figure.
 5. Lab x at stage x-angle: test (a), the foci calibration (frame_from_focus_pairs), test (b) with its U,
    recorded at lab 0 .. 165 deg in steps of 15 deg.
 6. Optional, if the freshness is still above 0.9: pairs of foci designed linear BEFORE the arm along stage phi and
    phi + 90 deg (frame of the speckle estimate of the first run), phi = -4, 0, 4, 8, 12 deg. For each pair the two
    extinction angles (2 deg scans, upward and downward). The pair whose two directions stay 90 deg apart behind
    the arm lies along the axes of the arm.

Focus power = sum over the 3x3 window at the brightest pixel near the target (located once per pattern), dark
level (median outside the NA disk) removed. The polarization direction of a focus behind the arm is its
extinction angle + 90 deg. Before the arm: field components along the arm axes divided by sqrt(1 + d), sqrt(1 - d).
Output: a new folder tm_focus_outputs/polarizer_followup_<date>_<time>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import polarizer_arm_estimator as pae          # noqa: E402
import run_polarization_axes as rpa            # noqa: E402
import run_polarizer_test as rpt               # noqa: E402

FINE_EXPOSURES = (24.0, 192.0, 768.0)
LOW_RANGE = tuple(float(a) for a in range(-10, 21, 2))
HIGH_RANGE = tuple(float(a) for a in range(80, 111, 2))
FULL_SCAN = tuple(float(a) for a in range(0, 180, 5))
EXTRA_DISPLACEMENT_ANGLES = (80.0, 96.0, 110.0, 170.0, 186.0, 200.0, 350.0, 6.0, 20.0)      # stage angles, in this order
PAIR_PHIS = (-4.0, 0.0, 4.0, 8.0, 12.0)
PAIR_HALF_RANGE = 14.0
UP_DOWN_LIMIT_DEG = 0.5
N_BOOT = 2000


# ---------------------------------------------------------------------------------------
# guard against a one-frame lag
# ---------------------------------------------------------------------------------------
class FrameGuard:
    """Mixin for a rig: one warm-up frame, discarded, after every change of stage angle, SLM pattern or exposure."""
    _dirty = True
    _last_exposure = None
    n_warmup = 0

    def move(self, angle_deg: float) -> None:
        super().move(angle_deg)
        self._dirty = True

    def show(self, mask_small_u8: np.ndarray) -> None:
        super().show(mask_small_u8)
        self._dirty = True

    def grab_fourier(self, exposure: float) -> np.ndarray:
        e = max(float(exposure), self.min_exposure)
        if self._dirty or e != self._last_exposure:
            super().grab_fourier(e)                      # warm-up frame, discarded
            self.n_warmup += 1
        self._dirty, self._last_exposure = False, e
        return super().grab_fourier(e)


class GuardedRig(FrameGuard, rpt.Rig):
    pass


# ---------------------------------------------------------------------------------------
# displacement circle
# ---------------------------------------------------------------------------------------
def circle_fit(angles_deg: Sequence[float], shifts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.radians(np.asarray(angles_deg, dtype=float))
    x = np.stack([np.ones_like(t), np.cos(t), np.sin(t)], axis=1)
    coef = np.linalg.lstsq(x, np.asarray(shifts, dtype=float), rcond=None)[0]
    resid = np.hypot(*(np.asarray(shifts) - x @ coef).T)
    return coef, resid


def circle_shift(coef: np.ndarray, angle_deg: float) -> np.ndarray:
    t = np.radians(angle_deg)
    return np.array([1.0, np.cos(t), np.sin(t)]) @ coef


def crop_offset(rig, coef: np.ndarray, angle_deg: float) -> Tuple[int, int]:
    d = circle_shift(coef, angle_deg) * rig.cam_per_grid
    return int(round(d[0])), int(round(d[1]))


def saturated_near(cropped: np.ndarray, yx: Tuple[int, int], cam_per_grid: float, half_grid: int = 3) -> bool:
    y0, y1 = int((yx[0] - half_grid) * cam_per_grid), int((yx[0] + half_grid + 1) * cam_per_grid)
    x0, x1 = int((yx[1] - half_grid) * cam_per_grid), int((yx[1] + half_grid + 1) * cam_per_grid)
    return bool(cropped[max(y0, 0):y1, max(x0, 0):x1].max() >= rpt.SATURATION)


# ---------------------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------------------
def record_series(rig, labels: Sequence[float], coef: np.ndarray, exposures: Sequence[float], plus180: bool = False) -> Dict:
    """
    Frames of the pattern now on the SLM at the analyzer angles ``labels`` (deg), recorded in the given order.
    Stage angle = label (modulo 360), or label + 180 deg (the same analyzer) if plus180.
    Returned sorted by label.
    """
    frames = {e: [] for e in exposures}
    sat = {e: {"A": [], "B": []} for e in exposures}
    sat_any = {e: False for e in exposures}
    for lab in labels:
        st = (lab + 180.0) if plus180 else (lab % 360.0)
        rig.move(st)
        off = crop_offset(rig, coef, st)
        for e in exposures:
            raw = rig.grab_fourier(e)
            crop = rig.crop(raw, off)
            frames[e].append(rig.to_grid(crop))
            sat[e]["A"].append(saturated_near(crop, rpt.TARGET_A, rig.cam_per_grid))
            sat[e]["B"].append(saturated_near(crop, rpt.TARGET_B, rig.cam_per_grid))
            sat_any[e] |= bool(raw.max() >= rpt.SATURATION)
    order = np.argsort(labels)
    return {"labels": [float(labels[i]) for i in order], "plus180": bool(plus180),
            "direction": "up" if labels[0] < labels[-1] else "down",
            "frames": {e: np.stack(frames[e])[order] for e in exposures},
            "saturated": {e: {f: [bool(sat[e][f][i]) for i in order] for f in ("A", "B")} for e in exposures},
            "saturated_anywhere": sat_any}


def up_down(rig, labels: Sequence[float], coef: np.ndarray, exposures: Sequence[float], plus180: bool = False) -> Tuple[Dict, Dict]:
    labels = sorted(labels)
    if plus180 or labels[0] >= 0:
        return record_series(rig, labels, coef, exposures, plus180), record_series(rig, labels[::-1], coef, exposures, plus180)
    # a range that crosses stage 0 without the 180 deg shift: negative labels are stage 350..358
    return record_series(rig, labels, coef, exposures), record_series(rig, labels[::-1], coef, exposures)


# ---------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------
def window_powers(frames: np.ndarray, support: np.ndarray, peak_yx: Tuple[int, int], half: int) -> np.ndarray:
    n = frames.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    win = (np.abs(yy - peak_yx[0]) <= half) & (np.abs(xx - peak_yx[1]) <= half)
    dark = np.array([np.median(f[~support]) for f in frames])
    return np.array([(frames[i][win] - dark[i]).sum() for i in range(len(frames))])


def locate_peak(frames: np.ndarray, support: np.ndarray, target: Tuple[int, int], radius: int = 3) -> Tuple[int, int]:
    n = frames.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    dark = np.array([np.median(f[~support]) for f in frames])
    total = (frames - dark[:, None, None]).sum(axis=0)
    near = (np.abs(yy - target[0]) <= radius) & (np.abs(xx - target[1]) <= radius)
    py, px = np.unravel_index(int(np.argmax(np.where(near, total, -np.inf))), total.shape)
    return int(py), int(px)


def _min_angle(c: np.ndarray) -> float:
    return float(np.degrees(0.5 * np.arctan2(-c[2], -c[1])))


def extinction_fit(angles_deg: Sequence[float], powers: Sequence[float], seed: int = 0) -> Dict:
    """Minimum of P(angle) = c0 + c1 cos(2 angle) + c2 sin(2 angle); angles in degrees."""
    a = np.asarray(angles_deg, dtype=float)
    p = np.asarray(powers, dtype=float)
    t = np.radians(a)
    x = np.stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t)], axis=1)
    c, *_ = np.linalg.lstsq(x, p, rcond=None)
    resid = p - x @ c
    s2 = float((resid ** 2).sum() / max(len(p) - 3, 1))
    cov = s2 * np.linalg.inv(x.T @ x)
    r2 = c[1] ** 2 + c[2] ** 2
    jac = np.array([0.0, -0.5 * c[2] / r2, 0.5 * c[1] / r2])
    sigma_fit = float(np.degrees(np.sqrt(jac @ cov @ jac)))
    centre = float(np.mean(a))
    theta = (_min_angle(c) - centre + 90.0) % 180.0 - 90.0 + centre             # the minimum nearest to the scanned range
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(p), len(p))
        if len(np.unique(i)) < 5:
            continue
        cb = np.linalg.lstsq(x[i], p[i], rcond=None)[0]
        boot.append((_min_angle(cb) - theta + 90.0) % 180.0 - 90.0 + theta)
    amp = float(np.hypot(c[1], c[2]))
    return {"extinction_angle_deg": float(theta), "sigma_fit_deg": sigma_fit, "sigma_bootstrap_deg": float(np.std(boot)),
            "minimum_over_maximum_of_fit": float((c[0] - amp) / (c[0] + amp)), "rms_residual_over_amplitude": float(np.sqrt(np.mean(resid ** 2)) / amp),
            "lowest_measured_power": float(p.min()), "angle_of_lowest_measured_power_deg": float(a[int(np.argmin(p))])}


def series_extinction(series: Dict, focus: str, peak: Tuple[int, int], support: np.ndarray) -> Dict:
    """Extinction angle of one focus in one series: main value at the longest unsaturated exposure, 3x3 window."""
    labels = np.array(series["labels"])
    exposures = sorted(series["frames"])
    fits = {}
    for e in exposures:
        if any(series["saturated"][e][focus]):
            continue
        for half in (1, 2):
            p = window_powers(series["frames"][e], support, peak, half) * (exposures[0] / e)
            fits[f"exposure_{e:g}_window_{2 * half + 1}"] = dict(extinction_fit(labels, p), powers=p.tolist())
    keys3 = [k for k in fits if k.endswith("window_3")]
    if not keys3:
        raise RuntimeError(f"Focus {focus} saturates at every exposure.")
    best = max(keys3, key=lambda k: float(k.split("_")[1]))
    main = fits[best]
    variants = np.array([v["extinction_angle_deg"] for v in fits.values()])
    return {"extinction_angle_deg": main["extinction_angle_deg"], "sigma_statistical_deg": max(main["sigma_fit_deg"], main["sigma_bootstrap_deg"]),
            "spread_over_exposures_and_windows_deg": float(np.ptp(variants)), "exposure_used": float(best.split("_")[1]),
            "distance_to_scan_edge_deg": float(min(main["extinction_angle_deg"] - labels.min(), labels.max() - main["extinction_angle_deg"])),
            "minimum_over_maximum_of_fit": main["minimum_over_maximum_of_fit"], "direction": series["direction"], "fits": fits}


def combine_up_down(up: Dict, down: Dict) -> Dict:
    diff = down["extinction_angle_deg"] - up["extinction_angle_deg"]
    stat = 0.5 * float(np.hypot(up["sigma_statistical_deg"], down["sigma_statistical_deg"]))
    spread = max(up["spread_over_exposures_and_windows_deg"], down["spread_over_exposures_and_windows_deg"])
    return {"extinction_angle_deg": 0.5 * (up["extinction_angle_deg"] + down["extinction_angle_deg"]),
            "up_deg": up["extinction_angle_deg"], "down_deg": down["extinction_angle_deg"], "down_minus_up_deg": float(diff),
            "up_down_agree": bool(abs(diff) <= UP_DOWN_LIMIT_DEG), "sigma_statistical_deg": stat,
            "spread_over_exposures_and_windows_deg": float(spread),
            "sigma_deg": float(np.sqrt(stat ** 2 + (0.5 * diff) ** 2 + (0.5 * spread) ** 2)),
            "exposure_used": [up["exposure_used"], down["exposure_used"]],
            "distance_to_scan_edge_deg": min(up["distance_to_scan_edge_deg"], down["distance_to_scan_edge_deg"]),
            "minimum_over_maximum_of_fit": [up["minimum_over_maximum_of_fit"], down["minimum_over_maximum_of_fit"]]}


def direction_before_arm(direction_deg: float, arm_axis_deg: float, d: float) -> float:
    """Linear polarization direction before the arm, from its direction behind the arm (stage scale, degrees)."""
    phi = np.radians(direction_deg - arm_axis_deg)
    before = np.arctan2(np.sin(phi) / np.sqrt(1.0 - d), np.cos(phi) / np.sqrt(1.0 + d))
    return float(np.degrees(before) + arm_axis_deg)


def angle_between(a_deg: float, b_deg: float) -> float:
    """Angle from direction a to direction b, modulo 180 deg (0..180)."""
    return float((b_deg - a_deg) % 180.0)


def directions_report(ext_a: float, sig_a: float, ext_b: float, sig_b: float, d_arm: float, arm_axes: Sequence[float]) -> Dict:
    da, db = (ext_a + 90.0) % 180.0, (ext_b + 90.0) % 180.0
    res = {"behind_arm": {"direction_A_deg": da, "direction_B_deg": db, "sigma_A_deg": sig_a, "sigma_B_deg": sig_b,
                          "angle_between_deg": angle_between(da, db), "sigma_deg": float(np.hypot(sig_a, sig_b))}, "before_arm": {}}
    for ax in arm_axes:
        ba, bb = direction_before_arm(da, ax, d_arm), direction_before_arm(db, ax, d_arm)
        eps = 0.05
        ga = (direction_before_arm(da + eps, ax, d_arm) - direction_before_arm(da - eps, ax, d_arm)) / (2 * eps)
        gb = (direction_before_arm(db + eps, ax, d_arm) - direction_before_arm(db - eps, ax, d_arm)) / (2 * eps)
        between, s = angle_between(ba, bb), float(np.hypot(ga * sig_a, gb * sig_b))
        res["before_arm"][f"arm_axes_at_{ax:g}"] = {"d": d_arm, "direction_A_deg": float(ba % 180.0), "direction_B_deg": float(bb % 180.0),
                                                    "sigma_A_deg": float(abs(ga) * sig_a), "sigma_B_deg": float(abs(gb) * sig_b),
                                                    "angle_between_deg": between, "sigma_deg": s, "deviation_from_90_deg": float(between - 90.0),
                                                    "ninety_within_2_sigma": bool(abs(between - 90.0) <= 2 * s)}
    return res


def print_directions(rep: Dict) -> None:
    b = rep["behind_arm"]
    print(f"  directions behind the arm: A {b['direction_A_deg']:.2f} +- {b['sigma_A_deg']:.2f} deg, B {b['direction_B_deg']:.2f} +- {b['sigma_B_deg']:.2f} deg; "
          f"angle between them {b['angle_between_deg']:.2f} +- {b['sigma_deg']:.2f} deg")
    for key, v in rep["before_arm"].items():
        print(f"  before the arm, {key.replace('_', ' ')} deg, d = {v['d']:+.2f}: A {v['direction_A_deg']:.2f} +- {v['sigma_A_deg']:.2f} deg, "
              f"B {v['direction_B_deg']:.2f} +- {v['sigma_B_deg']:.2f} deg; angle between them {v['angle_between_deg']:.2f} +- {v['sigma_deg']:.2f} deg "
              f"(deviation from 90 deg: {v['deviation_from_90_deg']:+.2f} deg)")


def power_at(series: Dict, focus: str, label: float, peak: Tuple[int, int], support: np.ndarray, best_exposure: bool) -> float:
    """Focus power at one analyzer angle, in units of the shortest exposure."""
    i = series["labels"].index(float(label))
    exposures = sorted(series["frames"])
    use = exposures[0]
    if best_exposure:
        ok = [e for e in exposures if not series["saturated"][e][focus][i]]
        use = max(ok) if ok else exposures[0]
    return float(window_powers(series["frames"][use][i:i + 1], support, peak, 1)[0] * exposures[0] / use)


def pass_fractions(low: Sequence[Dict], high: Sequence[Dict], peaks: Dict, support: np.ndarray, x: float, best_exposure: bool) -> Dict:
    """tau = P_x / (P_x + P_y) of both foci for the analyzer pair (x, x + 90), mean of the upward and downward scans."""
    p = {f: {"x": float(np.mean([power_at(s, f, x, peaks[f], support, best_exposure) for s in low])),
             "y": float(np.mean([power_at(s, f, x + 90.0, peaks[f], support, best_exposure) for s in high]))} for f in ("A", "B")}
    ta, tb = p["A"]["x"] / (p["A"]["x"] + p["A"]["y"]), p["B"]["x"] / (p["B"]["x"] + p["B"]["y"])
    return {"analyzer_pair_deg": [x, x + 90.0], "tau_A": ta, "tau_B": tb, "separation": ta - tb, "powers": p}


def save_series(path: Path, named: Dict[str, Dict]) -> None:
    out = {}
    for name, s in named.items():
        exposures = sorted(s["frames"])
        out[f"{name}__labels"] = np.array(s["labels"])
        out[f"{name}__exposures"] = np.array(exposures)
        out[f"{name}__frames"] = np.stack([s["frames"][e] for e in exposures]).astype(np.float32)
        out[f"{name}__saturated_A"] = np.array([s["saturated"][e]["A"] for e in exposures])
        out[f"{name}__saturated_B"] = np.array([s["saturated"][e]["B"] for e in exposures])
    np.savez_compressed(path, **out)


def full_scan_figure(path: Path, labels: np.ndarray, pa: np.ndarray, pb: np.ndarray, fa: Dict, fb: Dict, exposure: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fine = np.linspace(0, 180, 361)
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
        for ax, log in zip(axes, (False, True)):
            for p, f, colour, name in ((pa, fa, "tab:blue", "focus A (x')"), (pb, fb, "tab:red", "focus B (y')")):
                t = np.radians(fine)
                curve = f["c"][0] + f["c"][1] * np.cos(2 * t) + f["c"][2] * np.sin(2 * t)
                ax.plot(labels, p, "o", color=colour, ms=4, label=name)
                ax.plot(fine, np.clip(curve, 1e-3, None) if log else curve, "-", color=colour, lw=0.9)
            ax.set_ylabel("focus power (counts, 3x3 window)")
            if log:
                ax.set_yscale("log")
                ax.set_ylim(max(0.3, 0.5 * min(pa[pa > 0].min(), pb[pb > 0].min())), None)
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"Pattern (a): power of each focus against the polarizer angle (exposure {exposure:g}); lines: Malus fits", fontsize=9)
        axes[1].set_xlabel("polarizer angle on the stage scale (deg)")
        axes[1].set_xticks(np.arange(0, 181, 15))
        fig.tight_layout()
        fig.savefig(str(path), dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"  (figure not saved: {exc})")


def malus_fit(labels: np.ndarray, p: np.ndarray) -> Dict:
    t = np.radians(labels)
    x = np.stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t)], axis=1)
    c = np.linalg.lstsq(x, p, rcond=None)[0]
    amp = float(np.hypot(c[1], c[2]))
    return {"c": c.tolist(), "azimuth_deg": float(np.degrees(0.5 * np.arctan2(c[2], c[1])) % 180.0), "degree_of_linear_polarization": amp / float(c[0]),
            "rms_residual_over_amplitude": float(np.sqrt(np.mean((p - x @ c) ** 2)) / amp)}


# ---------------------------------------------------------------------------------------
# procedure
# ---------------------------------------------------------------------------------------
def run_followup(rig, tm_w, data: Dict, first_run: Path, out_dir: Path, x_angle: float, d_arm: float,
                 only_freshness: bool = False, do_pairs: bool = True) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = rig.n
    support, fit = rpt.disk(n, data["support_radius"]), rpt.disk(n, rpt.FIT_RADIUS)
    exposures = sorted({max(e, rig.min_exposure) for e in FINE_EXPOSURES})
    summary: Dict = {"started": datetime.now().isoformat(timespec="seconds"), "dataset_time": data["dataset_time"], "first_run": str(first_run),
                     "x_angle_deg": x_angle, "d_arm_for_the_extinction_analysis": d_arm, "b_of_weighted_tm": data["b"],
                     "exposures": exposures, "frame_guard": "one warm-up frame after every change of stage angle, SLM pattern or exposure"}

    def save() -> None:
        summary["warm_up_frames_so_far"] = int(getattr(rig, "n_warmup", 0))
        with open(out_dir / "polarizer_followup_summary.json", "w", encoding="utf-8") as f:
            json.dump(rpa._to_jsonable(summary), f, indent=1)

    def stop(reason: str) -> Dict:
        print(f"\nSTOPPED: {reason}")
        summary["stopped"] = reason
        rig.move(0.0)
        save()
        return summary

    def measure(angle: float) -> Dict:
        return rpt.measure_angle(rig, angle % 360.0, data["masks"][:rpt.N_SCAN], data["dataset_frames"][:rpt.N_SCAN], data["dataset_envelope"],
                                 data["tm_prediction"][:rpt.N_SCAN], rig.speckle_exposure, with_outline=False)

    first_3a = json.load(open(first_run / "polarizer_test_summary.json"))["3a"]["per_angle"]

    def freshness(tag: str) -> Tuple[float, Dict, Dict]:
        r0, r90 = measure(0.0), measure(90.0)
        ssum = r0["frames"] + r90["frames"]
        ref = data["dataset_frames"][:rpt.N_SCAN]
        pcc = [rpa.pcc(ssum[k][fit], ref[k][fit]) for k in range(rpt.N_FIRST)]
        lag = [rpa.pcc(ssum[k][fit], ref[k - 1][fit]) for k in range(1, len(ssum))]
        res = {"pcc_first_8": pcc, "mean": float(np.mean(pcc)), "mean_against_the_previous_mask": float(np.mean(lag)),
               "minutes_since_dataset": data["minutes_since_dataset"](), "time": datetime.now().isoformat(timespec="seconds"), "per_angle": {}}
        print(f"  (I_0 + I_90) against the dataset frames, 8 masks: {[round(float(v), 3) for v in pcc]}, mean {res['mean']:.3f}; "
              f"against the dataset frame of the previous mask (a one-frame lag would make this high): {res['mean_against_the_previous_mask']:.3f}; "
              f"{res['minutes_since_dataset']:.0f} min since the dataset")
        for ang, r in ((0.0, r0), (90.0, r90)):
            fr, old = r["frames"], first_3a[f"{ang:g}"]
            d = {"mean_level": r["info"]["mean_level"], "mean_level_first_run": old["mean_level"], "shift_px": r["info"]["shift_from_dataset_frames_px"],
                 "shift_first_run_px": old["shift_from_dataset_frames_px"],
                 "correlation_between_different_masks": float(np.mean([rpa.pcc(fr[k][fit], fr[k + 1][fit]) for k in range(len(fr) - 1)]))}
            res["per_angle"][f"{ang:g}"] = d
            print(f"  stage {ang:g} deg: mean level {d['mean_level']:.1f} (first run {d['mean_level_first_run']:.1f}); displacement "
                  f"({d['shift_px'][0]:+.2f}, {d['shift_px'][1]:+.2f}) px (first run ({d['shift_first_run_px'][0]:+.2f}, {d['shift_first_run_px'][1]:+.2f}))")
        np.savez_compressed(out_dir / f"freshness_frames_{tag}.npz", frames_stage_0=r0["frames"].astype(np.float32),
                            frames_stage_90=r90["frames"].astype(np.float32), dataset_frames=ref.astype(np.float32))
        return res["mean"], res, {0.0: r0, 90.0: r90}

    # ---- 1. freshness
    print("\n=== 1. Freshness ===")
    fresh, res, r = freshness("start")
    summary["freshness_start"] = res
    save()
    if only_freshness:
        return stop("only the freshness check was requested")
    if fresh < 0.9:
        print("  below 0.9: measuring once more")
        fresh, res, r = freshness("start_second_attempt")
        summary["freshness_start_second_attempt"] = res
        save()
        if fresh < 0.9:
            return stop("freshness below 0.9 twice; nothing else was recorded")

    # ---- 2. displacement circle
    print("\n=== 2. Displacement ===")
    angles = [float(k) for k in first_3a]
    shifts = [first_3a[k]["shift_from_dataset_frames_px"] for k in first_3a]
    new = {0.0: r[0.0]["info"]["shift_from_dataset_frames_px"], 90.0: r[90.0]["info"]["shift_from_dataset_frames_px"]}
    for ang in EXTRA_DISPLACEMENT_ANGLES:
        new[ang] = measure(ang)["info"]["shift_from_dataset_frames_px"]
    coef, resid = circle_fit(angles + list(new), np.array(shifts + list(new.values())))
    coef_first, _ = circle_fit(angles, np.array(shifts))
    pred_err = {a: float(np.hypot(*(np.array(new[a]) - circle_shift(coef_first, a)))) for a in new}
    summary["displacement"] = {"new_measurements_px": {f"{a:g}": v for a, v in new.items()},
                               "new_minus_circle_of_first_run_px": {f"{a:g}": v for a, v in pred_err.items()},
                               "circle_coefficients": coef.tolist(), "rms_residual_px": float(np.sqrt(np.mean(resid ** 2))), "largest_residual_px": float(resid.max()),
                               "radius_px": float(np.sqrt(0.5 * (coef[1] ** 2 + coef[2] ** 2).sum()))}
    print(f"  {len(new)} new angles against the circle of the first run: largest difference {max(pred_err.values()):.2f} px "
          f"(at stage {max(pred_err, key=pred_err.get):g} deg); circle through all {len(angles) + len(new)} measurements: radius "
          f"{summary['displacement']['radius_px']:.2f} px, rms residual {summary['displacement']['rms_residual_px']:.3f} px, largest {resid.max():.3f} px")
    save()

    # ---- 3. fine scans of pattern (a)
    print("\n=== 3. Fine scans of pattern (a), 2 deg steps, upward and downward ===")
    tags = {"x'/y'": "probe_xy_weighted_tm", "+45'/-45'": "probe_pm45_weighted_tm", "R'/L'": "probe_RL_weighted_tm"}
    u8 = {t: np.load(first_run / f"two_foci_{t}_phase_uint8.npy") for t in tags.values()}
    rig.show(u8["probe_xy_weighted_tm"])
    low_up, low_dn = up_down(rig, LOW_RANGE, coef, exposures)
    low180_up, low180_dn = up_down(rig, LOW_RANGE, coef, exposures, plus180=True)
    high_dn = record_series(rig, sorted(HIGH_RANGE)[::-1], coef, exposures)
    high_up = record_series(rig, sorted(HIGH_RANGE), coef, exposures)
    series = {"low_up": low_up, "low_down": low_dn, "high_up": high_up, "high_down": high_dn, "low_plus180_up": low180_up, "low_plus180_down": low180_dn}
    save_series(out_dir / "fine_scans_pattern_a.npz", series)
    e_low = exposures[0]
    all24 = np.concatenate([s["frames"][e_low] for s in (low_up, low_dn, high_up, high_dn)])
    peaks = {f: locate_peak(all24, support, t) for f, t in (("A", rpt.TARGET_A), ("B", rpt.TARGET_B))}
    dim_low = {f: bool(window_powers(low_up["frames"][e_low], support, peaks[f], 1).mean() < window_powers(high_up["frames"][e_low], support, peaks[f], 1).mean())
               for f in ("A", "B")}
    ext: Dict = {"peaks_yx": {f: list(peaks[f]) for f in peaks}, "foci": {}}
    for f in ("A", "B"):
        up_s, dn_s = (low_up, low_dn) if dim_low[f] else (high_up, high_dn)
        up_r, dn_r = series_extinction(up_s, f, peaks[f], support), series_extinction(dn_s, f, peaks[f], support)
        comb = combine_up_down(up_r, dn_r)
        comb.update({"extinction_range": "low" if dim_low[f] else "high", "up": up_r, "down": dn_r})
        if dim_low[f]:
            u2, d2 = series_extinction(low180_up, f, peaks[f], support), series_extinction(low180_dn, f, peaks[f], support)
            c2 = combine_up_down(u2, d2)
            comb["at_stage_plus_180"] = {"extinction_angle_deg": c2["extinction_angle_deg"], "up_deg": c2["up_deg"], "down_deg": c2["down_deg"],
                                         "minus_direct_deg": c2["extinction_angle_deg"] - comb["extinction_angle_deg"]}
        ext["foci"][f] = comb
        print(f"  focus {f} ({comb['extinction_range']} range): extinction upward {comb['up_deg']:+.2f} deg, downward {comb['down_deg']:+.2f} deg "
              f"(difference {comb['down_minus_up_deg']:+.2f} deg); mean {comb['extinction_angle_deg']:+.2f} +- {comb['sigma_deg']:.2f} deg "
              f"(statistical {comb['sigma_statistical_deg']:.2f}, spread over exposures and windows {comb['spread_over_exposures_and_windows_deg']:.2f}); "
              f"exposure {comb['exposure_used'][0]:g}; minimum / maximum of the fit {comb['minimum_over_maximum_of_fit'][0]:.5f}")
        for key, v in up_r["fits"].items():
            print(f"      upward, {key}: {v['extinction_angle_deg']:+.2f} deg (fit {v['sigma_fit_deg']:.2f}, bootstrap {v['sigma_bootstrap_deg']:.2f})")
        if "at_stage_plus_180" in comb:
            a = comb["at_stage_plus_180"]
            print(f"      the same range at stage angle + 180 deg: upward {a['up_deg']:+.2f}, downward {a['down_deg']:+.2f}, mean {a['extinction_angle_deg']:+.2f} deg "
                  f"({a['minus_direct_deg']:+.2f} deg from the direct scan)")
    fa, fb = ext["foci"]["A"], ext["foci"]["B"]
    ext.update(directions_report(fa["extinction_angle_deg"], fa["sigma_deg"], fb["extinction_angle_deg"], fb["sigma_deg"], d_arm, (0.0, x_angle)))
    print_directions(ext)
    low_s, high_s = (low_up, low_dn), (high_up, high_dn)
    ext["pass_fractions"] = {}
    for x in (0.0, x_angle):
        for label, best in (("one_exposure", False), ("longest_unsaturated_exposure_per_focus", True)):
            pf = pass_fractions(low_s, high_s, peaks, support, x, best)
            ext["pass_fractions"][f"pair_{x:g}_{x + 90:g}_{label}"] = pf
        p1, p2 = ext["pass_fractions"][f"pair_{x:g}_{x + 90:g}_one_exposure"], ext["pass_fractions"][f"pair_{x:g}_{x + 90:g}_longest_unsaturated_exposure_per_focus"]
        print(f"  pattern (a), analyzer pair {x:g} / {x + 90:g} deg, from the fine scan: A passes {p1['tau_A']:.4f}, B passes {p1['tau_B']:.4f}, separation {p1['separation']:+.4f} "
              f"(exposure {e_low:g}); with the longest unsaturated exposure for each focus: A {p2['tau_A']:.4f}, B {p2['tau_B']:.4f}, separation {p2['separation']:+.4f}")
    ext["minutes_since_dataset"], ext["time"] = data["minutes_since_dataset"](), datetime.now().isoformat(timespec="seconds")
    summary["extinction_pattern_a"] = ext
    save()
    if not (fa["up_down_agree"] and fb["up_down_agree"]):
        return stop(f"upward and downward extinction angles of pattern (a) differ by more than {UP_DOWN_LIMIT_DEG} deg")
    plus180_ok = all(abs(ext["foci"][f]["at_stage_plus_180"]["minus_direct_deg"]) <= UP_DOWN_LIMIT_DEG for f in ("A", "B") if "at_stage_plus_180" in ext["foci"][f])
    summary["plus_180_equivalent_within_limit"] = bool(plus180_ok)

    # ---- 4. full scan of pattern (a)
    print("\n=== 4. Full scan of pattern (a), 0..175 deg in steps of 5 deg ===")
    rig.show(u8["probe_xy_weighted_tm"])
    full = record_series(rig, FULL_SCAN, coef, [e_low])
    labels = np.array(full["labels"])
    pa, pb = window_powers(full["frames"][e_low], support, peaks["A"], 1), window_powers(full["frames"][e_low], support, peaks["B"], 1)
    ma, mb = malus_fit(labels, pa), malus_fit(labels, pb)
    np.savez_compressed(out_dir / "full_scan_pattern_a.npz", angles_deg=labels, frames=full["frames"][e_low].astype(np.float32), exposure=e_low,
                        power_A=pa, power_B=pb)
    full_scan_figure(out_dir / "full_scan_pattern_a.png", labels, pa, pb, ma, mb, e_low)
    summary["full_scan_pattern_a"] = {"angles_deg": labels.tolist(), "power_A": pa.tolist(), "power_B": pb.tolist(), "exposure": e_low,
                                      "saturated": bool(full["saturated_anywhere"][e_low]), "malus_fit_A": ma, "malus_fit_B": mb,
                                      "minutes_since_dataset": data["minutes_since_dataset"]()}
    print(f"  exposure {e_low:g}, saturation: {full['saturated_anywhere'][e_low]}; Malus fit of all 36 angles: A azimuth {ma['azimuth_deg']:.1f} deg, degree of linear "
          f"polarization {ma['degree_of_linear_polarization']:.3f} (residual / amplitude {ma['rms_residual_over_amplitude']:.3f}); B azimuth {mb['azimuth_deg']:.1f} deg, "
          f"{mb['degree_of_linear_polarization']:.3f} ({mb['rms_residual_over_amplitude']:.3f}); maximum of A {pa.max():.0f} at {labels[int(np.argmax(pa))]:g} deg, "
          f"of B {pb.max():.0f} at {labels[int(np.argmax(pb))]:g} deg")
    save()

    # ---- 5. lab x at stage x_angle: test (a), foci calibration, test (b)
    print(f"\n=== 5. Lab x at stage {x_angle:g} deg: test (a), foci calibration, test (b) ===")
    lab = (x_angle, x_angle + 90.0)
    rec = rpt.record_patterns(rig, {t: u8[t] for t in tags.values()}, lab, {a: crop_offset(rig, coef, a) for a in lab})
    metrics = {}
    for t in tags.values():
        rec[t]["frames"] = {0.0: rec[t]["frames"][lab[0]], 90.0: rec[t]["frames"][lab[1]]}          # keys: angle relative to lab x
        metrics[t] = rpt.foci_metrics(rec[t], support)
        rpt.save_frames(out_dir, f"x{x_angle:g}_{t}", rec[t])
    powers = {name: metrics[t]["powers_A_x_y_B_x_y"] for name, t in tags.items()}
    est = pae.frame_from_focus_pairs(powers)
    p_k, d_k = np.array(est["p"]), np.array(est["d_per_basis"])
    u_k = 0.5 * ((p_k[:, 0] - d_k) / (1 - d_k * p_k[:, 0]) - (p_k[:, 1] - d_k) / (1 - d_k * p_k[:, 1]))
    w_k = 1.0 - u_k ** 2
    use = w_k > 0.2
    d_con = float((w_k[use] * d_k[use]).sum() / w_k[use].sum()) if use.any() else float(est["d"])
    r1 = np.array(est["R1"])
    checks = {"R1_norm_within_0p9_1p1": bool(0.9 <= est["R1_norm"] <= 1.1), "pair_imbalance_below_0p15": bool(np.max(np.abs(est["pair_imbalance"])) < 0.15),
              "d_per_basis_agree_within_0p15": bool(use.sum() < 2 or np.ptp(d_k[use]) <= 0.15)}
    for t in tags.values():
        metrics[t]["before_arm"] = rpt.before_arm(metrics[t]["powers_A_x_y_B_x_y"], est["d"])
        rpt.print_foci(t, metrics[t])
    ang_minus = float(np.degrees(np.arccos(np.clip(-r1[0], -1, 1))))
    print(f"  (in the lines above, 'stage 0' and 'stage 90' mean lab x and lab y, that is stage {lab[0]:g} and {lab[1]:g} deg)")
    print(f"  frame_from_focus_pairs: d = {est['d']:+.3f} (per basis {[round(v, 3) for v in est['d_per_basis']]}; from the pairs that constrain d: {d_con:+.3f}); "
          f"learned b = {data['b']:+.3f}")
    print(f"  R1 = ({r1[0]:+.3f}, {r1[1]:+.3f}, {r1[2]:+.3f}), |R1| before normalization {est['R1_norm']:.3f}; angle to (-1, 0, 0): {ang_minus:.1f} deg; "
          f"pair imbalance {[round(v, 3) for v in est['pair_imbalance']]}; checks {checks}")
    summary["calibration_x_angle"] = {"d": est["d"], "d_per_basis": est["d_per_basis"], "d_from_constraining_pairs": d_con, "R1": est["R1"], "R1_norm": est["R1_norm"],
                                      "pair_imbalance": est["pair_imbalance"], "angle_R1_to_minus_e1_deg": ang_minus, "checks": checks, "powers": powers,
                                      "patterns": metrics, "U_real": est["U"].real.tolist(), "U_imag": est["U"].imag.tolist(),
                                      "minutes_since_dataset": data["minutes_since_dataset"](), "time": datetime.now().isoformat(timespec="seconds")}
    np.save(out_dir / f"jones_matrix_U_from_foci_x{x_angle:g}.npy", est["U"])
    save()
    tag_b = f"x{x_angle:g}_lab_axes_U_from_foci"
    pattern_b = rpt.save_pattern(rig, out_dir, tag_b, rpt.design_pattern(tm_w, est["U"], support))
    stage_angles = [x_angle + a for a in rpt.SCAN_ANGLES]
    rec_b = rpt.record_patterns(rig, {tag_b: pattern_b}, stage_angles, {a: crop_offset(rig, coef, a) for a in stage_angles})[tag_b]
    rec_b["frames"] = {float(a - x_angle): f for a, f in rec_b["frames"].items()}
    mb6 = rpt.foci_metrics(rec_b, support, est["d"])
    rpt.save_frames(out_dir, tag_b, rec_b)
    rpt.print_foci(tag_b, mb6)
    for name in ("A", "B"):
        print(f"      scan, focus {name}: azimuth {mb6[name]['azimuth_deg']:.1f} deg from lab x, degree of linear polarization {mb6[name]['degree_of_linear_polarization']:.3f}")
    mb6["minutes_since_dataset"], mb6["time"] = data["minutes_since_dataset"](), datetime.now().isoformat(timespec="seconds")
    summary["test_b_x_angle"] = mb6
    save()

    # ---- 6. optional: pairs linear before the arm along phi and phi + 90
    if do_pairs:
        print("\n=== 6. Optional: pairs of foci linear before the arm along stage phi and phi + 90 deg ===")
        fresh2, res2, _ = freshness("before_pairs")
        summary["freshness_before_pairs"] = res2
        save()
        if fresh2 < 0.9:
            return stop("freshness below 0.9 before the optional pairs; they were not recorded")
        first_frames = dict(zip(rpt.MAIN_ANGLES, np.load(first_run / "speckle_frames_main_angles.npz")["frames"]))
        est_c = rpt.step_c(first_frames, data, {})["_est_all"]
        frame = {"U": est_c["U"], "d": est_c["d"], "theta0_deg": est_c["theta0_deg"]}
        pairs: Dict = {"frame_of_the_design": {"d": frame["d"], "theta0_deg": frame["theta0_deg"]}, "phi": {}}
        summary["pairs"] = pairs
        for phi in PAIR_PHIS:
            tag = f"pair_before_arm_phi{phi:+g}"
            print(f"  phi = {phi:+g} deg: designing")
            pat = rpt.save_pattern(rig, out_dir, tag, rpt.design_pattern(tm_w, rpt.projection_design(frame, (phi, phi + 90.0), with_arm=False), support))
            pred = rpt.predicted_scan(tm_w, pat, frame, rpt.SCAN_ANGLES)
            rig.show(pat)
            rec_s, windows = {}, {}
            for f in ("A", "B"):
                c = (pred[f]["azimuth_deg"] + 90.0 + 45.0) % 180.0 - 45.0                      # predicted extinction angle, -45 .. 135
                c = 2.0 * round(c / 2.0)
                labels = [c + k for k in np.arange(-PAIR_HALF_RANGE, PAIR_HALF_RANGE + 0.1, 2.0)]
                plus180 = bool(c < 45.0)                                                       # ranges near 0 are scanned at stage angle + 180
                windows[f] = {"centre_deg": float(c), "plus180": plus180, "predicted_extinction_deg": float((pred[f]["azimuth_deg"] + 90.0) % 180.0)}
                rec_s[f] = up_down(rig, labels, coef, exposures, plus180=plus180)
            save_series(out_dir / f"fine_scans_{tag}.npz", {f"{f}_{d}": rec_s[f][i] for f in ("A", "B") for i, d in ((0, "up"), (1, "down"))})
            other = {"A": "B", "B": "A"}
            res_phi: Dict = {"windows": windows, "foci": {}}
            for f in ("A", "B"):
                pk = locate_peak(np.concatenate([rec_s[other[f]][0]["frames"][e_low], rec_s[f][0]["frames"][e_low]]), support, rpt.TARGET_A if f == "A" else rpt.TARGET_B)
                comb = combine_up_down(series_extinction(rec_s[f][0], f, pk, support), series_extinction(rec_s[f][1], f, pk, support))
                comb["peak_yx"] = list(pk)
                res_phi["foci"][f] = comb
            a_, b_ = res_phi["foci"]["A"], res_phi["foci"]["B"]
            res_phi.update(directions_report(a_["extinction_angle_deg"], a_["sigma_deg"], b_["extinction_angle_deg"], b_["sigma_deg"], d_arm, (0.0, x_angle)))
            res_phi["minutes_since_dataset"] = data["minutes_since_dataset"]()
            pairs["phi"][f"{phi:g}"] = res_phi
            bh = res_phi["behind_arm"]
            print(f"      extinction A {a_['extinction_angle_deg']:+.2f} +- {a_['sigma_deg']:.2f} deg (up {a_['up_deg']:+.2f}, down {a_['down_deg']:+.2f}; predicted "
                  f"{windows['A']['predicted_extinction_deg']:.1f}), B {b_['extinction_angle_deg']:+.2f} +- {b_['sigma_deg']:.2f} deg (up {b_['up_deg']:+.2f}, down {b_['down_deg']:+.2f}; "
                  f"predicted {windows['B']['predicted_extinction_deg']:.1f}); directions behind the arm A {bh['direction_A_deg']:.2f}, B {bh['direction_B_deg']:.2f} deg; "
                  f"angle between them {bh['angle_between_deg']:.2f} +- {bh['sigma_deg']:.2f} deg; distance of the minima to the scan edge {a_['distance_to_scan_edge_deg']:.0f}, "
                  f"{b_['distance_to_scan_edge_deg']:.0f} deg")
            save()
            if not (a_["up_down_agree"] and b_["up_down_agree"]):
                return stop(f"upward and downward extinction angles of the pair phi = {phi:g} deg differ by more than {UP_DOWN_LIMIT_DEG} deg")
        # the pair that stays 90 deg apart
        phis = np.array([float(k) for k in pairs["phi"]])
        between = np.array([pairs["phi"][k]["behind_arm"]["angle_between_deg"] for k in pairs["phi"]])
        dir_a = np.array([pairs["phi"][k]["behind_arm"]["direction_A_deg"] for k in pairs["phi"]])
        dir_b = np.array([pairs["phi"][k]["behind_arm"]["direction_B_deg"] for k in pairs["phi"]])
        dir_a = (dir_a + 90.0) % 180.0 - 90.0                                                  # A lies near 0: use -90 .. 90
        sig = np.array([pairs["phi"][k]["behind_arm"]["sigma_deg"] for k in pairs["phi"]])
        slope, icpt = np.polyfit(phis, between, 1)
        phi_star = float((90.0 - icpt) / slope)
        line_a, line_b = np.polyfit(phis, dir_a, 1), np.polyfit(phis, dir_b, 1)
        axes_star = [float(np.polyval(line_a, phi_star)), float(np.polyval(line_b, phi_star))]
        resid_fit = float(np.sqrt(np.mean((between - (slope * phis + icpt)) ** 2)))
        sig_phi = float(max(np.mean(sig), resid_fit) / abs(slope) / np.sqrt(len(phis)) * np.sqrt(1 + (phi_star - phis.mean()) ** 2 / phis.var()))
        pairs["result"] = {"phi_deg": phis.tolist(), "angle_between_behind_arm_deg": between.tolist(), "sigma_deg": sig.tolist(),
                           "direction_A_behind_arm_deg": dir_a.tolist(), "direction_B_behind_arm_deg": dir_b.tolist(),
                           "linear_fit_slope": float(slope), "linear_fit_intercept": float(icpt), "rms_residual_of_linear_fit_deg": resid_fit,
                           "phi_with_90_deg": phi_star, "sigma_phi_deg": sig_phi, "arm_axes_stage_deg": axes_star,
                           "arm_axes_sigma_deg": float(sig_phi * max(abs(line_a[0]), abs(line_b[0]))),
                           "inside_scanned_phi_range": bool(phis.min() <= phi_star <= phis.max())}
        print(f"  angle between the two directions behind the arm against phi: {dict(zip([float(p) for p in phis], [round(float(v), 2) for v in between]))}")
        print(f"  linear fit: slope {slope:+.3f} deg per deg, 90 deg at phi = {phi_star:+.2f} +- {sig_phi:.2f} deg (rms residual {resid_fit:.2f} deg); "
              f"axes of the arm at stage {axes_star[0]:+.2f} and {axes_star[1]:.2f} deg")
        save()

    rig.move(0.0)
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    print(f"\nSaved everything to {out_dir}; warm-up frames discarded: {getattr(rig, 'n_warmup', 0)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weighted-tm", required=True)
    ap.add_argument("--first-run", required=True, help="output folder of run_polarizer_test.py (displacements, SLM patterns, speckle frames)")
    ap.add_argument("--h5", default=str(TM_LIB / "measurement_focus.h5"))
    ap.add_argument("--x-angle", type=float, default=6.0, help="stage angle that defines lab x")
    ap.add_argument("--d-arm", type=float, default=-0.70, help="diattenuation used to divide out the arm in the extinction analysis")
    ap.add_argument("--only-freshness", action="store_true", help="record the freshness frames and their diagnostics, then stop")
    ap.add_argument("--no-pairs", action="store_true", help="skip the optional pairs of step 6")
    args = ap.parse_args()

    from run_tm_focus_optimization import PipelineConfig, TMFocusPipeline      # hardware libraries are loaded here

    pipeline = TMFocusPipeline(PipelineConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tm_w = rpt.load_tm(Path(args.weighted_tm), device)
    data = rpt.prepare_data(tm_w, Path(args.h5), Path(pipeline.cfg.optimized_masks_h5_path), pipeline.support_radius_px("fourier"))
    print(f"Weighted TM: {args.weighted_tm} (b = {data['b']:+.4f}); dataset of {data['dataset_time']}, {data['minutes_since_dataset']():.0f} min old")
    out_dir = Path(pipeline.cfg.output_dir) / f"polarizer_followup_{datetime.now():%Y%m%d_%H%M}"
    stage = rpa.RotatingPolarizerStage()
    stage.connect()
    try:
        pipeline.setup()
        try:
            from create_refs import get_camera_exposure_minimum
            rig = GuardedRig(pipeline, stage, float(get_camera_exposure_minimum(pipeline.cameras[1])))
            run_followup(rig, tm_w, data, Path(args.first_run), out_dir, args.x_angle, args.d_arm, only_freshness=args.only_freshness,
                         do_pairs=not args.no_pairs)
        finally:
            pipeline.teardown()
    finally:
        stage.close()


if __name__ == "__main__":
    main()
