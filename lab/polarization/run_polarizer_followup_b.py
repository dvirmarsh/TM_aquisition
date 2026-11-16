"""
Repair of step 5 of run_polarizer_followup.py (15 Nov 2026): test (b) with lab x at stage x-angle.

In the follow-up run, polarizer_arm_estimator.frame_from_focus_pairs returned d = NaN: with lab x at stage
6 deg, focus A of the x'/y' pair is blocked so completely that its measured power at lab x is zero, the pair
is exactly pure, and the estimator's per-pair formula divides 0 by 0. The matrix U was invalid, so the
pattern of test (b) of that run is invalid too.

This script
 1. recomputes the calibration from the probe frames ALREADY recorded in the follow-up folder, with the same
    formulas but a common d: d is taken from the pairs that constrain it (weight 1 - u^2 > 0.2, evaluated with
    the common d, iterated), then u_A, u_B and R1 of all three pairs follow with that d;
 2. checks the freshness (stops below 0.9, after a second attempt);
 3. designs test (b) with that U and records it at lab 0 .. 165 deg in steps of 15 deg.

    python run_polarizer_followup_b.py --weighted-tm <checkpoint> --run-dir <follow-up folder> --first-run <first run folder>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import polarizer_arm_estimator as pae          # noqa: E402
import run_polarization_axes as rpa            # noqa: E402
import run_polarizer_followup as rpf           # noqa: E402
import run_polarizer_test as rpt               # noqa: E402

TAGS = {"x'/y'": "probe_xy_weighted_tm", "+45'/-45'": "probe_pm45_weighted_tm", "R'/L'": "probe_RL_weighted_tm"}


def robust_frame_from_focus_pairs(powers: Dict) -> Dict:
    """frame_from_focus_pairs with a common d (see the module docstring). powers: {basis: (P_A_x, P_A_y, P_B_x, P_B_y)}."""
    names = list(pae.PROBE_ROTATIONS)
    p = np.array([[(powers[n][0] - powers[n][1]) / (powers[n][0] + powers[n][1]),
                   (powers[n][2] - powers[n][3]) / (powers[n][2] + powers[n][3])] for n in names])
    p = np.clip(p, -0.9999, 0.9999)
    d_k = np.array([pae.diattenuation_from_orthogonal_pair(pa, pb) for pa, pb in p])
    d = float(np.median(d_k))
    use = np.ones(len(names), dtype=bool)
    for _ in range(5):
        u_a, u_b = (p[:, 0] - d) / (1 - d * p[:, 0]), (p[:, 1] - d) / (1 - d * p[:, 1])
        w = 1.0 - (0.5 * (u_a - u_b)) ** 2
        use = w > 0.2
        if not use.any():
            break
        d = float(np.clip((w[use] * d_k[use]).sum() / w[use].sum(), -0.98, 0.98))
    u_a, u_b = (p[:, 0] - d) / (1 - d * p[:, 0]), (p[:, 1] - d) / (1 - d * p[:, 1])
    r1 = 0.5 * (u_a - u_b)
    r = pae.rotation_with_first_row(r1)
    return {"d": d, "d_per_basis": d_k.tolist(), "pairs_used_for_d": [n for n, f in zip(names, use) if f],
            "R1": (r1 / np.linalg.norm(r1)).tolist(), "R1_norm": float(np.linalg.norm(r1)), "pair_imbalance": (u_a + u_b).tolist(),
            "p": p.tolist(), "R": r, "U": pae.jones_from_rotation(r)}


def calibration_from_saved_frames(run_dir: Path, x_angle: float, support: np.ndarray) -> Dict:
    metrics, powers = {}, {}
    for name, t in TAGS.items():
        z = np.load(run_dir / f"two_foci_x{x_angle:g}_{t}_frames.npz")
        rec = {"exposure": float(z["exposure"]), "saturated": False, "frames": {float(a): f for a, f in zip(z["angles_deg"], z["frames"])}}
        metrics[t] = rpt.foci_metrics(rec, support)
        powers[name] = metrics[t]["powers_A_x_y_B_x_y"]
    est = robust_frame_from_focus_pairs(powers)
    d_use = np.array([dk for dk, n in zip(est["d_per_basis"], pae.PROBE_ROTATIONS) if n in est["pairs_used_for_d"]])
    est["checks"] = {"R1_norm_within_0p9_1p1": bool(0.9 <= est["R1_norm"] <= 1.1),
                     "pair_imbalance_below_0p15": bool(np.max(np.abs(est["pair_imbalance"])) < 0.15),
                     "d_per_basis_agree_within_0p15": bool(len(d_use) < 2 or np.ptp(d_use) <= 0.15)}
    for t in metrics:
        metrics[t]["before_arm"] = rpt.before_arm(metrics[t]["powers_A_x_y_B_x_y"], est["d"])
    est["patterns"], est["powers"] = metrics, powers
    return est


def print_calibration(est: Dict, b: float) -> None:
    r1 = np.array(est["R1"])
    ang_minus = float(np.degrees(np.arccos(np.clip(-r1[0], -1, 1))))
    est["angle_R1_to_minus_e1_deg"] = ang_minus
    for t, m in est["patterns"].items():
        rpt.print_foci(t, m)
    print("  (in the lines above, 'stage 0' and 'stage 90' mean lab x and lab y)")
    print(f"  d = {est['d']:+.3f} from the pairs {est['pairs_used_for_d']} (per basis {[round(v, 3) for v in est['d_per_basis']]}); learned b = {b:+.3f}")
    print(f"  R1 = ({r1[0]:+.3f}, {r1[1]:+.3f}, {r1[2]:+.3f}), |R1| before normalization {est['R1_norm']:.3f}; angle to (-1, 0, 0): {ang_minus:.1f} deg; "
          f"pair imbalance {[round(v, 3) for v in est['pair_imbalance']]}; checks {est['checks']}")


def run_b(rig, tm_w, data: Dict, run_dir: Path, first_run: Path, x_angle: float) -> Dict:
    n = rig.n
    support, fit = rpt.disk(n, data["support_radius"]), rpt.disk(n, rpt.FIT_RADIUS)
    summary = json.load(open(run_dir / "polarizer_followup_summary.json"))
    coef = np.array(summary["displacement"]["circle_coefficients"])
    res: Dict = {"started": datetime.now().isoformat(timespec="seconds")}

    def save() -> None:
        with open(run_dir / "polarizer_followup_b_summary.json", "w", encoding="utf-8") as f:
            json.dump(rpa._to_jsonable(res), f, indent=1)

    print("\n=== Calibration with foci, recomputed from the saved probe frames ===")
    est = calibration_from_saved_frames(run_dir, x_angle, support)
    print_calibration(est, data["b"])
    res["calibration_x_angle"] = {k: v for k, v in est.items() if k not in ("R", "U")}
    res["calibration_x_angle"].update({"U_real": est["U"].real.tolist(), "U_imag": est["U"].imag.tolist()})
    np.save(run_dir / f"jones_matrix_U_from_foci_x{x_angle:g}_v2.npy", est["U"])
    save()
    if not np.all(np.isfinite(est["U"])):
        print("\nSTOPPED: the recomputed U is not finite.")
        res["stopped"] = "U not finite"
        save()
        return res

    print("\n=== Freshness ===")
    for attempt in (1, 2):
        frames = {}
        for ang in (0.0, 90.0):
            frames[ang] = rpt.measure_angle(rig, ang, data["masks"][:rpt.N_SCAN], data["dataset_frames"][:rpt.N_SCAN], data["dataset_envelope"],
                                            data["tm_prediction"][:rpt.N_SCAN], rig.speckle_exposure, with_outline=False)["frames"]
        ssum = frames[0.0] + frames[90.0]
        pcc = [rpa.pcc(ssum[k][fit], data["dataset_frames"][k][fit]) for k in range(rpt.N_FIRST)]
        lag = [rpa.pcc(ssum[k][fit], data["dataset_frames"][k - 1][fit]) for k in range(1, len(ssum))]
        fresh = float(np.mean(pcc))
        res[f"freshness_attempt_{attempt}"] = {"pcc_first_8": pcc, "mean": fresh, "mean_against_the_previous_mask": float(np.mean(lag)),
                                               "minutes_since_dataset": data["minutes_since_dataset"]()}
        print(f"  (I_0 + I_90) against the dataset frames, 8 masks: mean {fresh:.3f}; against the previous mask: {np.mean(lag):.3f}; "
              f"{data['minutes_since_dataset']():.0f} min since the dataset")
        save()
        if fresh >= 0.9:
            break
    if fresh < 0.9:
        print("\nSTOPPED: freshness below 0.9 twice; test (b) was not recorded.")
        res["stopped"] = "freshness below 0.9 twice"
        rig.move(0.0)
        save()
        return res

    print(f"\n=== Test (b) with lab x at stage {x_angle:g} deg ===")
    tag_b = f"x{x_angle:g}_lab_axes_U_from_foci_v2"
    pattern_b = rpt.save_pattern(rig, run_dir, tag_b, rpt.design_pattern(tm_w, est["U"], support))
    stage_angles = [x_angle + a for a in rpt.SCAN_ANGLES]
    rec_b = rpt.record_patterns(rig, {tag_b: pattern_b}, stage_angles, {a: rpf.crop_offset(rig, coef, a) for a in stage_angles})[tag_b]
    rig.move(0.0)
    rec_b["frames"] = {float(a - x_angle): f for a, f in rec_b["frames"].items()}
    mb = rpt.foci_metrics(rec_b, support, est["d"])
    rpt.save_frames(run_dir, tag_b, rec_b)
    rpt.print_foci(tag_b, mb)
    for name in ("A", "B"):
        print(f"      scan, focus {name}: azimuth {mb[name]['azimuth_deg']:.1f} deg from lab x, degree of linear polarization {mb[name]['degree_of_linear_polarization']:.3f}; "
              f"power per angle {[round(v) for v in mb[name]['window_power']]}")
    mb["minutes_since_dataset"], mb["time"] = data["minutes_since_dataset"](), datetime.now().isoformat(timespec="seconds")
    res["test_b_x_angle"] = mb
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    print(f"\nSaved to {run_dir}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weighted-tm", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--first-run", required=True)
    ap.add_argument("--h5", default=str(TM_LIB / "measurement_focus.h5"))
    ap.add_argument("--x-angle", type=float, default=6.0)
    args = ap.parse_args()

    from run_tm_focus_optimization import PipelineConfig, TMFocusPipeline

    pipeline = TMFocusPipeline(PipelineConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tm_w = rpt.load_tm(Path(args.weighted_tm), device)
    data = rpt.prepare_data(tm_w, Path(args.h5), Path(pipeline.cfg.optimized_masks_h5_path), pipeline.support_radius_px("fourier"))
    stage = rpa.RotatingPolarizerStage()
    stage.connect()
    try:
        pipeline.setup()
        try:
            from create_refs import get_camera_exposure_minimum
            rig = rpf.GuardedRig(pipeline, stage, float(get_camera_exposure_minimum(pipeline.cameras[1])))
            run_b(rig, tm_w, data, Path(args.run_dir), Path(args.first_run), args.x_angle)
        finally:
            pipeline.teardown()
    finally:
        stage.close()


if __name__ == "__main__":
    main()
