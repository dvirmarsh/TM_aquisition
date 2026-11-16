"""
Dark level of the Fourier-plane camera at short exposure, from the saved 2 deg scans of pattern (a) (offline).

    python dark_level_analysis.py <fine_scans_pattern_a.npz> <output json>

The scans hold, per analyzer angle, frames at exposures 24, 192 and 768 recorded one after the other (Mono12,
128 x 128 grid, counts). For every angle:
  background      = pixels inside the NA disk (radius 58.09 px around (63.5, 63.5)) outside the two 3 x 3 focus windows
                    (second variant: outside two 7 x 7 windows); saturated pixels are left out
  mean background = mean count of those pixels, scaled to exposure 24 (x 24 / exposure)
  PBR             = peak pixel of the transmitted focus / mean background
      (i)  peak and background from the exposure-24 frame
      (ii) peak from the exposure-24 frame, background from the exposure-768 frame scaled by 24 / 768
  implied offset  = background (768, scaled) - background (24), counts per pixel at exposure 24
Transmitted focus: B in the range -10 .. 20 deg (A is near extinction), A in the range 80 .. 110 deg.
Also: response of single pixels, the exposure-24 count against the exposure-768 count scaled to exposure 24.
"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1])
N, RADIUS, SAT = 128, 58.094796018360924, 4095
yy, xx = np.mgrid[:N, :N]
disk = (yy - (N - 1) / 2.0) ** 2 + (xx - (N - 1) / 2.0) ** 2 <= RADIUS ** 2
TARGETS = {"A": (48, 56), "B": (80, 72)}


def peak_near(frame, yx, search=3):
    near = (np.abs(yy - yx[0]) <= search) & (np.abs(xx - yx[1]) <= search)
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(np.where(near, frame, -np.inf))), frame.shape))


def lab_index(series, lab):
    return int(np.nonzero(z[f"{series}__labels"] == lab)[0][0])


# focus positions: brightest pixel near the target in the exposure-24 frame where that focus is transmitted
peaks = {"A": peak_near(z["high_up__frames"][0, lab_index("high_up", 96.0)], TARGETS["A"]),
         "B": peak_near(z["low_up__frames"][0, lab_index("low_up", 6.0)], TARGETS["B"])}


def windows(half):
    m = np.zeros((N, N), dtype=bool)
    for py, px in peaks.values():
        m |= (np.abs(yy - py) <= half) & (np.abs(xx - px) <= half)
    return m


bg_mask = {3: disk & ~windows(1), 7: disk & ~windows(3)}
series_focus = {"low_up": "B", "low_down": "B", "high_up": "A", "high_down": "A", "low_plus180_up": "B", "low_plus180_down": "B"}
out = {"peaks_yx": peaks, "n_background_pixels": {str(k): int(v.sum()) for k, v in bg_mask.items()}, "series": {}}
resp_true, resp_read = [], []
for series, focus in series_focus.items():
    labels, exposures, frames = z[f"{series}__labels"], z[f"{series}__exposures"], z[f"{series}__frames"]
    rows = []
    for i, lab in enumerate(labels):
        row = {"analyzer_deg": float(lab), "transmitted_focus": focus}
        f24 = frames[0, i]
        row["peak_24"] = float(f24[peaks[focus]])
        for win, mask in bg_mask.items():
            r = {}
            for k, e in enumerate(exposures):
                f = frames[k, i]
                ok = mask & (f < SAT)
                r[f"bg_mean_exp{e:g}_scaled_to_24"] = float(f[ok].mean() * 24.0 / e)
                r[f"bg_mean_exp{e:g}_counts"] = float(f[ok].mean())
                r[f"bg_zero_fraction_exp{e:g}"] = float((f[ok] == 0).mean())
                r[f"bg_saturated_pixels_excluded_exp{e:g}"] = int((mask & (f >= SAT)).sum())
            b24, b192, b768 = (r[f"bg_mean_exp{e:g}_scaled_to_24"] for e in exposures)
            r["pbr_i_exp24_alone"] = row["peak_24"] / b24 if b24 > 0 else float("inf")
            r["pbr_ii_peak24_bg768"] = row["peak_24"] / b768
            r["pbr_peak24_bg192"] = row["peak_24"] / b192
            r["ratio_i_over_ii"] = r["pbr_i_exp24_alone"] / r["pbr_ii_peak24_bg768"]
            r["implied_offset_counts_per_pixel"] = b768 - b24
            r["offset_192_vs_24"] = b192 - b24
            r["bg192_over_bg768_scaled"] = b192 / b768
            row[f"window_{win}x{win}"] = r
        rows.append(row)
        # single-pixel response, background pixels (7 x 7 exclusion), unsaturated at 768
        f768 = frames[2, i]
        ok = bg_mask[7] & (f768 < SAT)
        resp_true.append(f768[ok] * 24.0 / 768.0)
        resp_read.append(f24[ok])
    out["series"][series] = rows

resp_true, resp_read = np.concatenate(resp_true), np.concatenate(resp_read)
edges = [0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0]
resp = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (resp_true >= lo) & (resp_true < hi)
    if m.sum() >= 30:
        resp.append({"expected_counts_from_768": [lo, hi], "n_pixels": int(m.sum()), "mean_expected": float(resp_true[m].mean()),
                     "mean_read_at_24": float(resp_read[m].mean()), "zero_fraction_at_24": float((resp_read[m] == 0).mean()),
                     "read_minus_expected": float(resp_read[m].mean() - resp_true[m].mean())})
out["single_pixel_response_exp24_vs_exp768"] = resp
hi_sig = resp_true >= 8.0
c = np.polyfit(resp_true[hi_sig], resp_read[hi_sig], 1)
out["linear_fit_above_8_counts"] = {"slope": float(c[0]), "offset_counts": float(c[1]), "n_pixels": int(hi_sig.sum())}


def summary(series_names, win):
    vals = {}
    for key in ("bg_mean_exp24_scaled_to_24", "bg_mean_exp192_scaled_to_24", "bg_mean_exp768_scaled_to_24", "bg_zero_fraction_exp24",
                "bg_zero_fraction_exp192", "bg_zero_fraction_exp768", "pbr_i_exp24_alone", "pbr_ii_peak24_bg768", "ratio_i_over_ii",
                "implied_offset_counts_per_pixel", "bg192_over_bg768_scaled"):
        v = np.array([r[f"window_{win}x{win}"][key] for s in series_names for r in out["series"][s]])
        vals[key] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    return vals


out["summary"] = {f"{name}_window_{w}x{w}": summary(s, w)
                  for name, s in (("range_-10_20_B_transmitted", ["low_up", "low_down"]), ("range_80_110_A_transmitted", ["high_up", "high_down"]),
                                  ("range_-10_20_stage_plus_180_B_transmitted", ["low_plus180_up", "low_plus180_down"])) for w in (3, 7)}
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1)

print("focus positions:", peaks, " background pixels:", out["n_background_pixels"])
for series in ("low_up", "low_down", "high_up", "high_down"):
    lab = 6.0 if series.startswith("low") else 96.0
    r = out["series"][series][lab_index(series, lab)]
    for w in (3, 7):
        q = r[f"window_{w}x{w}"]
        print(f"{series:<10} analyzer {lab:g} deg, focus {r['transmitted_focus']}, peak(24) {r['peak_24']:g}, windows {w}x{w}: "
              f"bg scaled to 24: {q['bg_mean_exp24_scaled_to_24']:.4f} / {q['bg_mean_exp192_scaled_to_24']:.4f} / {q['bg_mean_exp768_scaled_to_24']:.4f} (exp 24/192/768); "
              f"zeros {q['bg_zero_fraction_exp24']:.3f} / {q['bg_zero_fraction_exp192']:.3f} / {q['bg_zero_fraction_exp768']:.3f}; "
              f"PBR (i) {q['pbr_i_exp24_alone']:.0f}, (ii) {q['pbr_ii_peak24_bg768']:.0f}, ratio {q['ratio_i_over_ii']:.2f}, offset {q['implied_offset_counts_per_pixel']:.3f}")
for name, v in out["summary"].items():
    print(f"{name}: bg24 {v['bg_mean_exp24_scaled_to_24']['mean']:.4f}, bg192 {v['bg_mean_exp192_scaled_to_24']['mean']:.4f}, bg768 {v['bg_mean_exp768_scaled_to_24']['mean']:.4f}; "
          f"zeros {v['bg_zero_fraction_exp24']['mean']:.3f}/{v['bg_zero_fraction_exp192']['mean']:.3f}/{v['bg_zero_fraction_exp768']['mean']:.3f}; "
          f"ratio (i)/(ii) {v['ratio_i_over_ii']['mean']:.2f} +- {v['ratio_i_over_ii']['std']:.2f} [{v['ratio_i_over_ii']['min']:.2f}, {v['ratio_i_over_ii']['max']:.2f}]; "
          f"offset {v['implied_offset_counts_per_pixel']['mean']:.3f} +- {v['implied_offset_counts_per_pixel']['std']:.3f}; bg192/bg768 {v['bg192_over_bg768_scaled']['mean']:.3f}")
print("single-pixel response (expected from 768 -> read at 24):")
for r in resp:
    print(f"  expected {r['expected_counts_from_768'][0]:>6g} .. {r['expected_counts_from_768'][1]:<6g} n {r['n_pixels']:>7}  mean expected {r['mean_expected']:8.3f}  "
          f"read {r['mean_read_at_24']:8.3f}  read - expected {r['read_minus_expected']:+7.3f}  zeros {r['zero_fraction_at_24']:.3f}")
print("linear fit above 8 counts:", out["linear_fit_above_8_counts"])
