"""
Offline analysis of the frames of run_pbr_bias_check.py: PBR with a background that is neither dark-biased nor
truncated by saturation.

    python pbr_bias_hdr_analysis.py <pbr_bias_check_<stamp> folder> [output json]

Why: PBR (ii) of the check takes the background from the longest exposure and leaves out the pixels that saturate
there. Those are the brightest background pixels (the halo of the focus), so leaving them out makes the background too
low. Here every pixel of the 128 x 128 frame gets the value of the longest exposure at which it is not affected by
saturation, scaled to E0 (an HDR composite of the five-frame means). The peak is always taken from the E0 frames.

Per target:
  PBR (i)            one E0 frame, support outside the 3 x 3 window (the paper's number), mean of five frames
  PBR_hdr_3x3        peak(E0) / mean of the composite over the support outside the 3 x 3 window   (same definition)
  PBR_hdr_7x7        the same outside the 7 x 7 window
  where the E0 background goes missing: ring between 3 x 3 and 7 x 7, pixels outside 7 x 7 that saturate at the longest
  exposure, and all other pixels
Also: response of single frame pixels at E0 against the composite built without the E0 frames, and the composite
built from the exposures up to 384 only (a check of the exposure scaling).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

run = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "pbr_bias_hdr_analysis.json"
s = json.load(open(run / "pbr_bias_check_summary.json", encoding="utf-8"))
z = np.load(run / "focus_frames_128.npz")
SAT, n = 4095, 128
e0 = float(s["e0"])
exposures = [float(e) for e in s["exposures"] if float(e) >= e0]
yy, xx = np.mgrid[:n, :n]
support = (yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2 <= float(s["support_radius_px"]) ** 2


def composite(mean_frames, sat, use):
    """Per pixel: the longest exposure in ``use`` at which the pixel is not affected by saturation, scaled to E0."""
    hdr = np.full((n, n), np.nan)
    src = np.zeros((n, n))
    for e in sorted(use):
        ok = ~sat[e]
        hdr[ok] = mean_frames[e][ok] * e0 / e
        src[ok] = e
    return hdr, src


res = {"e0": e0, "exposures": exposures, "targets": []}
resp_true, resp_read = [], []
for t in s["targets"]:
    k = t["index"]
    a = t["analysis"]
    py, px = a["peak_yx"]
    frames = {e: z[f"target_{k}_exposure_{e:g}"].astype(np.float64) for e in exposures}
    raw = np.load(run / f"target_{k}_raw_crops_uint16.npz")
    sat = {}
    for e in exposures:
        sat_raw = (raw[f"exposure_{e:g}"] >= SAT).any(axis=0).astype(np.float32)
        sat[e] = (np.flip(cv2.resize(sat_raw, (n, n), interpolation=cv2.INTER_LINEAR)) > 0) | (frames[e] >= SAT).any(axis=0)
    means = {e: frames[e].mean(axis=0) for e in exposures}
    hdr, src = composite(means, sat, exposures)
    hdr_384, _ = composite(means, sat, [e for e in exposures if e <= 384.0])
    hdr_no_e0, src_no_e0 = composite(means, sat, [e for e in exposures if e > e0])
    assert np.isfinite(hdr[support]).all()
    win3 = (np.abs(yy - py) <= 1) & (np.abs(xx - px) <= 1)
    win7 = (np.abs(yy - py) <= 3) & (np.abs(xx - px) <= 3)
    bg3, bg7 = support & ~win3, support & ~win7
    ring = support & win7 & ~win3
    e_long = max(exposures)
    sat_out7 = bg7 & sat[e_long]
    rest = bg7 & ~sat[e_long]
    f0 = means[e0]
    peak0 = float(a["peak_e0_counts_mean"])
    zones = {}
    for name, m in (("ring_3x3_to_7x7", ring), ("outside_7x7_saturated_at_longest", sat_out7), ("outside_7x7_other", rest), ("all_background_3x3", bg3)):
        zones[name] = {"pixels": int(m.sum()), "sum_e0_counts": float(f0[m].sum()), "sum_composite_counts": float(hdr[m].sum()),
                       "mean_e0": float(f0[m].mean()) if m.any() else None, "mean_composite": float(hdr[m].mean()) if m.any() else None,
                       "zero_fraction_e0_frames": float((frames[e0][:, m] == 0).mean()) if m.any() else None}
    r = {"index": k, "target_yx": t["target_yx"], "peak_yx": [py, px], "peak_e0_counts": peak0,
         "pbr_i": a["pbr_i"], "pbr_i_std": a["pbr_i_std"], "pbr_ii_as_specified": a.get("pbr_ii"), "pbr_iii_as_specified": a.get("pbr_iii"),
         "background_3x3_e0": float(f0[bg3].mean()), "background_3x3_composite": float(hdr[bg3].mean()),
         "background_7x7_e0": float(f0[bg7].mean()), "background_7x7_composite": float(hdr[bg7].mean()),
         "background_7x7_longest_without_saturated": float((means[e_long][rest] * e0 / e_long).mean()),
         "pbr_hdr_3x3": peak0 / float(hdr[bg3].mean()), "pbr_hdr_7x7": peak0 / float(hdr[bg7].mean()),
         "pbr_hdr_3x3_from_exposures_up_to_384": peak0 / float(hdr_384[bg3].mean()),
         "peak_composite_over_peak_e0": float(hdr_no_e0[py, px] / f0[py, px]) if np.isfinite(hdr_no_e0[py, px]) else None,
         "zones": zones,
         "pixels_by_source_exposure_in_background": {f"{e:g}": int((src[bg3] == e).sum()) for e in exposures}}
    r["ratio_i_over_hdr_3x3"] = r["pbr_i"] / r["pbr_hdr_3x3"]
    r["ratio_i_over_hdr_3x3_up_to_384"] = r["pbr_i"] / r["pbr_hdr_3x3_from_exposures_up_to_384"]
    r["pbr_e0_7x7"] = peak0 / r["background_7x7_e0"]
    r["ratio_e0_7x7_over_hdr_7x7"] = r["pbr_e0_7x7"] / r["pbr_hdr_7x7"]
    r["ratio_ii_as_specified_over_hdr_7x7"] = (r["pbr_ii_as_specified"] / r["pbr_hdr_7x7"]) if r["pbr_ii_as_specified"] else None
    r["missing_counts_per_background_pixel"] = r["background_3x3_composite"] - r["background_3x3_e0"]
    res["targets"].append(r)
    # response of single E0 frames against the composite without E0 (background pixels outside the 7 x 7 window)
    ok = bg7 & np.isfinite(hdr_no_e0)
    for f in frames[e0]:
        resp_true.append(hdr_no_e0[ok])
        resp_read.append(f[ok])

resp_true, resp_read = np.concatenate(resp_true), np.concatenate(resp_read)
edges = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0, 512.0]
table = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (resp_true >= lo) & (resp_true < hi)
    if m.sum() >= 30:
        table.append({"expected_counts": [lo, hi], "n": int(m.sum()), "mean_expected": float(resp_true[m].mean()), "mean_read_e0": float(resp_read[m].mean()),
                      "read_minus_expected": float(resp_read[m].mean() - resp_true[m].mean()), "zero_fraction_e0": float((resp_read[m] == 0).mean())})
res["single_pixel_response_e0"] = table
hi_sig = (resp_true >= 8.0) & (resp_true <= 512.0)
if hi_sig.sum() > 50:
    c = np.polyfit(resp_true[hi_sig], resp_read[hi_sig], 1)
    res["linear_fit_8_to_512_counts"] = {"slope": float(c[0]), "offset_counts": float(c[1]), "n": int(hi_sig.sum())}


def stat(key):
    v = np.array([t[key] for t in res["targets"] if t.get(key) is not None], dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


res["summary"] = {k: stat(k) for k in ("pbr_i", "pbr_hdr_3x3", "pbr_hdr_7x7", "ratio_i_over_hdr_3x3", "ratio_i_over_hdr_3x3_up_to_384", "ratio_e0_7x7_over_hdr_7x7",
                                       "ratio_ii_as_specified_over_hdr_7x7", "background_3x3_e0", "background_3x3_composite", "missing_counts_per_background_pixel",
                                       "peak_composite_over_peak_e0")}
json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)

print("target  peak   PBR(i)   PBR_hdr_3x3  (i)/hdr_3x3  [up to 384]  bg3 E0 -> composite   PBR E0 7x7 / hdr 7x7   (ii)/hdr_7x7   peak composite/E0")
for r in res["targets"]:
    print(f"  {r['index']}   {r['peak_e0_counts']:6.0f}  {r['pbr_i']:6.1f}   {r['pbr_hdr_3x3']:8.1f}     {r['ratio_i_over_hdr_3x3']:.3f}        {r['ratio_i_over_hdr_3x3_up_to_384']:.3f}      "
          f"{r['background_3x3_e0']:.3f} -> {r['background_3x3_composite']:.3f}      {r['pbr_e0_7x7']:7.1f} / {r['pbr_hdr_7x7']:6.1f} = {r['ratio_e0_7x7_over_hdr_7x7']:.3f}    "
          f"{r['ratio_ii_as_specified_over_hdr_7x7']:.3f}        {r['peak_composite_over_peak_e0']}")
for k, v in res["summary"].items():
    print(f"{k}: {v['mean']:.3f} +- {v['std']:.3f} ({v['min']:.3f} to {v['max']:.3f})")
print("zones, target 1:")
for name, v in res["targets"][0]["zones"].items():
    print(f"  {name:<34} {v['pixels']:>6} px  sum E0 {v['sum_e0_counts']:9.0f}  sum composite {v['sum_composite_counts']:9.0f}  mean E0 {v['mean_e0']:8.3f} composite {v['mean_composite']:8.3f}  zeros {v['zero_fraction_e0_frames']:.3f}")
print("source exposure of the background pixels, target 1:", res["targets"][0]["pixels_by_source_exposure_in_background"])
print("single-pixel response at E0 (expected from the longer exposures -> read):")
for r in table:
    print(f"  {r['expected_counts'][0]:>6g} .. {r['expected_counts'][1]:<6g} n {r['n']:>7}  expected {r['mean_expected']:8.3f}  read {r['mean_read_e0']:8.3f}  read - expected {r['read_minus_expected']:+7.3f}  zeros {r['zero_fraction_e0']:.3f}")
print("linear fit 8 .. 512 counts:", res.get("linear_fit_8_to_512_counts"))
