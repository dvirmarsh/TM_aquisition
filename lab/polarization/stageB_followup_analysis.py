"""
Offline analysis of a run of run_polarizer_followup.py (no hardware).

    python stageB_followup_analysis.py <follow-up folder>

1. Pattern (a): the two polarization directions before the arm for several positions of the arm axes.
2. Pairs: fit of the squeeze of the arm to the measured directions behind the arm,
       direction_A(phi) = theta + atan(k tan(phi + delta - theta)),
       direction_B(phi) = theta + 90 + atan(tan(phi + delta - theta) / k),   k = sqrt((1 - d) / (1 + d)),
   with theta (stage angle of the suppressed axis of the arm), d and delta (offset between the nominal and the
   actual direction before the arm) free. Uncertainties from a bootstrap over the measured directions.
3. Low-signal response of the camera: focus power at the shortest exposure against the same power at the
   longest exposure (scaled), for the focus that goes through extinction.
Writes followup_extra_analysis.json into the folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TM_LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(TM_LIB)); sys.path.insert(0, str(TM_LIB.parent))
import run_polarizer_followup as rpf           # noqa: E402

run = Path(sys.argv[1])
s = json.load(open(run / "polarizer_followup_summary.json"))
out = {}

# ---- 1. pattern (a) before the arm, for several arm axes
e = s["extinction_pattern_a"]
fa, fb = e["foci"]["A"], e["foci"]["B"]
print("Pattern (a): extinction A %+.2f +- %.2f deg, B %+.2f +- %.2f deg" % (fa["extinction_angle_deg"], fa["sigma_deg"], fb["extinction_angle_deg"], fb["sigma_deg"]))
rows = {}
for d in (-0.70,):
    for ax in (0.0, 3.0, 4.0, 4.87, 5.0, 6.0):
        rep = rpf.directions_report(fa["extinction_angle_deg"], fa["sigma_deg"], fb["extinction_angle_deg"], fb["sigma_deg"], d, (ax,))
        v = rep["before_arm"][f"arm_axes_at_{ax:g}"]
        rows[f"d{d:+.2f}_axes{ax:g}"] = v
        print(f"  d = {d:+.2f}, arm axes at stage {ax:g}: A {v['direction_A_deg']:.2f}, B {v['direction_B_deg']:.2f}, angle between {v['angle_between_deg']:.2f} +- {v['sigma_deg']:.2f} "
              f"(deviation from 90: {v['deviation_from_90_deg']:+.2f})")
grid = np.arange(-2.0, 10.01, 0.01)
dev = np.array([rpf.angle_between(rpf.direction_before_arm((fa["extinction_angle_deg"] + 90) % 180, ax, -0.70),
                                  rpf.direction_before_arm((fb["extinction_angle_deg"] + 90) % 180, ax, -0.70)) - 90.0 for ax in grid])
ax90 = float(grid[int(np.argmin(np.abs(dev)))])
print(f"  the two directions are exactly 90 deg apart before the arm for arm axes at stage {ax90:.2f} deg (d = -0.70)")
out["pattern_a_before_arm"] = {"rows": rows, "arm_axis_for_90_deg": ax90}

# ---- 2. pairs: fit of the squeeze
if "pairs" in s and "result" in s["pairs"]:
    pr = s["pairs"]["phi"]
    phi = np.array([float(k) for k in pr])
    da = np.array([(pr[k]["behind_arm"]["direction_A_deg"] + 90.0) % 180.0 - 90.0 for k in pr])
    db = np.array([pr[k]["behind_arm"]["direction_B_deg"] for k in pr])
    sa = np.array([pr[k]["behind_arm"]["sigma_A_deg"] for k in pr])
    sb = np.array([pr[k]["behind_arm"]["sigma_B_deg"] for k in pr])

    def model(par, phi_):
        theta, dd, delta = par
        k = np.sqrt((1 - dd) / (1 + dd))
        x = np.radians(phi_ + delta - theta)
        return theta + np.degrees(np.arctan(k * np.tan(x))), theta + 90.0 + np.degrees(np.arctan(np.tan(x) / k))

    def fit(da_, db_):
        best = None
        for th0 in np.arange(0.0, 10.1, 0.25):                       # coarse grid, then Gauss-Newton
            for d0 in (-0.6, -0.7, -0.8):
                par = np.array([th0, d0, 0.0])
                for _ in range(60):
                    ma, mb = model(par, phi)
                    r = np.concatenate([(da_ - ma) / sa, (db_ - mb) / sb])
                    jac = np.empty((len(r), 3))
                    for j, h in enumerate((1e-3, 1e-4, 1e-3)):
                        p2 = par.copy(); p2[j] += h
                        m2a, m2b = model(p2, phi)
                        jac[:, j] = np.concatenate([(m2a - ma) / sa, (m2b - mb) / sb]) / h
                    step = np.linalg.lstsq(jac, r, rcond=None)[0]
                    par = par + step
                    par[1] = np.clip(par[1], -0.97, -0.05)
                    if np.abs(step).max() < 1e-7:
                        break
                ma, mb = model(par, phi)
                chi2 = float((((da_ - ma) / sa) ** 2).sum() + (((db_ - mb) / sb) ** 2).sum())
                if best is None or chi2 < best[1]:
                    best = (par.copy(), chi2)
        return best

    par, chi2 = fit(da, db)
    rng = np.random.default_rng(0)
    boot = np.array([fit(da + sa * rng.normal(size=len(da)), db + sb * rng.normal(size=len(db)))[0] for _ in range(200)])
    ma, mb = model(par, phi)
    res = {"phi_deg": phi.tolist(), "direction_A_deg": da.tolist(), "direction_B_deg": db.tolist(), "model_A_deg": ma.tolist(), "model_B_deg": mb.tolist(),
           "theta_arm_deg": float(par[0]), "theta_arm_sigma_deg": float(boot[:, 0].std()), "d": float(par[1]), "d_sigma": float(boot[:, 1].std()),
           "delta_deg": float(par[2]), "delta_sigma_deg": float(boot[:, 2].std()), "chi2": chi2, "degrees_of_freedom": int(2 * len(phi) - 3),
           "rms_residual_deg": float(np.sqrt(np.mean(np.concatenate([da - ma, db - mb]) ** 2))),
           "linear_interpolation": s["pairs"]["result"]}
    print("Pairs: directions behind the arm, measured (model):")
    for i in range(len(phi)):
        print(f"  phi {phi[i]:+5.1f}: A {da[i]:+7.2f} ({ma[i]:+7.2f}), B {db[i]:6.2f} ({mb[i]:6.2f}); angle between {(db[i] - da[i]):.2f} ({(mb[i] - ma[i]):.2f})")
    print(f"  squeeze fit: arm axes at stage {par[0]:.2f} +- {boot[:, 0].std():.2f} deg and {par[0] + 90:.2f} deg; d = {par[1]:+.3f} +- {boot[:, 1].std():.3f}; "
          f"offset of the designed direction {par[2]:+.2f} +- {boot[:, 2].std():.2f} deg; chi2 {chi2:.1f} for {2 * len(phi) - 3} degrees of freedom; rms residual {res['rms_residual_deg']:.2f} deg")
    print(f"  phi with 90 deg (squeeze fit): {par[0] - par[2]:+.2f} deg; linear interpolation of the script: {s['pairs']['result']['phi_with_90_deg']:+.2f} deg, "
          f"axes {s['pairs']['result']['arm_axes_stage_deg'][0]:+.2f} and {s['pairs']['result']['arm_axes_stage_deg'][1]:.2f} deg")
    out["pairs_squeeze_fit"] = res

# ---- 3. low-signal response at the shortest exposure
low = {}
for f in ("A", "B"):
    fits = e["foci"][f]["up"]["fits"]
    keys = sorted([k for k in fits if k.endswith("window_3")], key=lambda k: float(k.split("_")[1]))
    p_short, p_long = np.array(fits[keys[0]]["powers"]), np.array(fits[keys[-1]]["powers"])
    c = np.polyfit(p_long, p_short, 1)
    low[f] = {"exposures": [keys[0], keys[-1]], "power_shortest": p_short.tolist(), "power_longest_scaled": p_long.tolist(),
              "slope": float(c[0]), "offset_counts": float(c[1])}
    print(f"Focus {f}, {keys[0]} against {keys[-1]} (scaled to the shortest exposure): P_short = {c[0]:.3f} * P_long {c[1]:+.1f} counts; "
          f"shortest {[round(v, 1) for v in p_short[:8]]} ..., longest {[round(v, 1) for v in p_long[:8]]} ...")
out["low_signal_response"] = low
json.dump(out, open(run / "followup_extra_analysis.json", "w"), indent=1)
print("Saved", run / "followup_extra_analysis.json")
