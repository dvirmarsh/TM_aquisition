"""
Stage B (revised), offline: collect the numbers of the polarizer test for the report.

    python stageB_collect_report_numbers.py <run folder>

Reads polarizer_test_summary.json of a run of run_polarizer_test.py; writes report_numbers.json there.
Adds: leaks of the blocked focus with the labels as they appear, separations before the arm for several d,
and, from the scans, the separation that each pattern would show if the polarizer axes were turned by a
common offset (Malus fit of each focus).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TM_LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(TM_LIB))
import polarizer_arm_estimator as pae          # noqa: E402

run = Path(sys.argv[1])
s = json.load(open(run / "polarizer_test_summary.json"))
b = s["b_of_weighted_tm"]
d_foci, d_foci2 = s["3d"]["d"], s["3d"]["d_from_constraining_pairs"]
d_speckle = s["3c"]["all"]["d"]
out = {"b": b, "d_foci": d_foci, "d_foci_constraining_pairs": d_foci2, "d_speckle": [s["3c"]["all"][k] for k in ("d", "d_from_s2_amplitude", "d_from_total_intensity")]}


def before(p, d):
    est = {"weights_xy": [1 + d, 1 - d]}
    ta, tb = pae.pass_fraction_before_arm(p[0], p[1], est), pae.pass_fraction_before_arm(p[2], p[3], est)
    return ta, tb, ta - tb


def describe(name, m):
    p = m["powers_A_x_y_B_x_y"]                       # A at 0, A at 90, B at 0, B at 90
    ta, tb = p[0] / (p[0] + p[1]), p[2] / (p[2] + p[3])
    swapped = ta < tb
    # blocked focus at stage 0 and at stage 90, with the labels as they appear
    leak0 = (p[0] / p[2]) if swapped else (p[2] / p[0])
    leak90 = (p[3] / p[1]) if swapped else (p[1] / p[3])
    r = {"tau_A": ta, "tau_B": tb, "separation_raw": ta - tb, "appears": "A at stage 90, B at stage 0" if swapped else "A at stage 0, B at stage 90",
         "leak_blocked_at_stage_0": leak0, "leak_blocked_at_stage_90": leak90,
         "pbr_A_0_90": m["pbr_A_at_0_90"], "pbr_B_0_90": m["pbr_B_at_0_90"], "exposure": m["exposure"], "saturated": m["saturated"],
         "peaks": [m["A"]["peak_yx"], m["B"]["peak_yx"]]}
    for label, d in (("d_foci", d_foci), ("d_minus_b", -abs(b)), ("d_speckle", d_speckle)):
        a_, b_, sep = before(p, d)
        r[f"before_arm_{label}"] = {"d": d, "tau_A": a_, "tau_B": b_, "separation": sep}
    print(f"{name}: {r['appears']}; raw A {ta:.3f}, B {tb:.3f}, separation {ta - tb:+.3f}; before the arm (d = {d_foci:+.3f}) "
          f"A {r['before_arm_d_foci']['tau_A']:.3f}, B {r['before_arm_d_foci']['tau_B']:.3f}, separation {r['before_arm_d_foci']['separation']:+.3f}; "
          f"with d = {-abs(b):+.3f}: {r['before_arm_d_minus_b']['separation']:+.3f}")
    print(f"    leak of the blocked focus: {leak0:.4f} at stage 0, {leak90:.4f} at stage 90; PBR A {m['pbr_A_at_0_90'][0]:.0f} / {m['pbr_A_at_0_90'][1]:.0f}, "
          f"B {m['pbr_B_at_0_90'][0]:.0f} / {m['pbr_B_at_0_90'][1]:.0f} (stage 0 / 90); exposure {m['exposure']:g}; peaks {r['peaks']}")
    return r


print("--- recorded at stage 0 and 90 deg ---")
out["a_weighted_xy"] = describe("(a) x'/y' of the weighted TM", s["3d"]["patterns"]["probe_xy_weighted_tm"])
out["b_U_from_foci"] = describe("(b) U from the foci calibration", s["3e_b"])
out["c_blind_xy"] = describe("(c) x'/y' of the blind TM", s["3d"]["patterns"]["xy_blind_tm"])


def scan(name, m):
    ang = np.radians(np.array(m["angles_deg"]))
    design = 0.5 * np.stack([np.ones_like(ang), np.cos(2 * ang), np.sin(2 * ang)], axis=1)
    c = {f: np.linalg.lstsq(design, np.array(m[f]["window_power"]), rcond=None)[0] for f in ("A", "B")}
    resid = {f: float(np.sqrt(np.mean((design @ c[f] - np.array(m[f]["window_power"])) ** 2)) / (0.5 * c[f][0])) for f in ("A", "B")}
    offs = np.arange(-20.0, 20.01, 0.5)
    sep = []
    for o in offs:
        t = np.radians(o)
        p = {f: (0.5 * (c[f][0] + c[f][1] * np.cos(2 * t) + c[f][2] * np.sin(2 * t)),
                 0.5 * (c[f][0] - c[f][1] * np.cos(2 * t) - c[f][2] * np.sin(2 * t))) for f in ("A", "B")}
        sep.append(p["A"][0] / sum(p["A"]) - p["B"][0] / sum(p["B"]))
    sep = np.array(sep)
    i = int(np.argmax(np.abs(sep)))
    r = {"azimuth_A": m["A"]["azimuth_deg"], "dolp_A": m["A"]["degree_of_linear_polarization"],
         "azimuth_B": m["B"]["azimuth_deg"], "dolp_B": m["B"]["degree_of_linear_polarization"],
         "fit_residual_rel": resid, "angle_between_foci_deg": float(abs((m["A"]["azimuth_deg"] - m["B"]["azimuth_deg"] + 90) % 180 - 90)),
         "separation_0_90": float(sep[list(offs).index(0.0)]), "best_offset_deg": float(offs[i]), "separation_at_best_offset": float(sep[i]),
         "separation_at_offset_6": float(sep[list(offs).index(6.0)])}
    if "predicted_with_speckle_frame" in m:
        r["predicted_azimuth_A_B"] = [m["predicted_with_speckle_frame"]["A"]["azimuth_deg"], m["predicted_with_speckle_frame"]["B"]["azimuth_deg"]]
    print(f"{name}: azimuth A {r['azimuth_A']:.1f} deg (DoLP {r['dolp_A']:.3f}), B {r['azimuth_B']:.1f} deg (DoLP {r['dolp_B']:.3f}); "
          f"fit residual {100 * resid['A']:.1f} % / {100 * resid['B']:.1f} %; raw separation {r['separation_0_90']:+.3f} at stage 0/90, "
          f"{r['separation_at_offset_6']:+.3f} at 6/96, largest {r['separation_at_best_offset']:+.3f} at offset {r['best_offset_deg']:+.1f} deg"
          + (f"; predicted azimuths {r['predicted_azimuth_A_B'][0]:.1f}, {r['predicted_azimuth_A_B'][1]:.1f}" if "predicted_azimuth_A_B" in r else ""))
    return r


print("--- scans 0..165 deg (behind the arm) ---")
out["scan_b"] = scan("(b) U from the foci calibration", s["3e_b"])
if "3f" in s:
    for tag, label in (("scan_probe_xy_weighted_tm", "(a) x'/y' of the weighted TM"), ("scan_xy_blind_tm", "(c) x'/y' of the blind TM"),
                       ("pm45_at_polarizer", "(f) +-45 at the polarizer"), ("pm45_before_arm", "(f) +-45 before the arm")):
        if tag in s["3f"]["patterns"]:
            out["scan_" + tag] = scan(label, s["3f"]["patterns"][tag])
    f = s["3f"]["patterns"]
    for tag in ("pm45_at_polarizer", "pm45_before_arm"):
        m = f[tag]
        q = m.get("predicted_with_speckle_frame", {})
        print(f"{tag}: A passes {m['tau45_A']:.3f} at stage 45, B {m['tau45_B']:.3f}; separation {m['separation_45_135']:+.3f} "
              f"(predicted {q['A']['tau_45_135'] - q['B']['tau_45_135']:+.3f}); leaks {m['leak_B_over_A_at_45']:.4f} at 45, {m['leak_A_over_B_at_135']:.4f} at 135; "
              f"PBR A {m['pbr_A_at_45_135'][0]:.0f} / {m['pbr_A_at_45_135'][1]:.0f}, B {m['pbr_B_at_45_135'][0]:.0f} / {m['pbr_B_at_45_135'][1]:.0f}")
        out[tag] = {k: m[k] for k in ("tau45_A", "tau45_B", "separation_45_135", "leak_B_over_A_at_45", "leak_A_over_B_at_135", "pbr_A_at_45_135", "pbr_B_at_45_135")}
        out[tag]["predicted_separation"] = q["A"]["tau_45_135"] - q["B"]["tau_45_135"]

print("--- times ---")
times = {"dataset": s["dataset_time"], "3a": s["3a"]["time"], "3b": s["3b"]["time"], "3b_minutes": s["3b"]["minutes_since_dataset"],
         "3d": s["3d"]["time"], "3d_minutes": s["3d"]["minutes_since_dataset"], "3e_b": s["3e_b"]["time"], "3e_b_minutes": s["3e_b"]["minutes_since_dataset"],
         "freshness_3b_first8": s["3b"]["pcc_first_8_mean"], "freshness_3b_all": s["3b"]["pcc_all_mean"]}
if "3f" in s:
    times.update({"3f": s["3f"]["time"], "3f_minutes": s["3f"]["minutes_since_dataset"], "freshness_3f_first8": s["3f"]["freshness_mean"]})
out["times"] = times
print(json.dumps(times, indent=1))
r1 = np.array(s["3d"]["R1"])
out["R1_as_state"] = {"azimuth_deg": float(np.degrees(0.5 * np.arctan2(r1[1], r1[0])) % 180.0), "ellipticity_deg": float(np.degrees(0.5 * np.arcsin(r1[2]))),
                      "cross_term_power": float((1 - abs(r1[0])) / 2)}
print("R1 as a model state: azimuth %.1f deg, ellipticity %+.1f deg; cross-term power %.3f" % tuple(out["R1_as_state"].values()))
json.dump(out, open(run / "report_numbers.json", "w"), indent=1)
print("Saved", run / "report_numbers.json")
