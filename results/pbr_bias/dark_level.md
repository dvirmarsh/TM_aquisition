# Dark level at short exposure (offline, from saved frames)

Written 15 Nov 2026. No hardware was used. Paths are relative to this folder.

## Result

- At exposure 24 a pixel of the Fourier-plane camera reads about **1.2 counts below a linear response and is clipped at 0**.
- The mean background of an exposure-24 frame is therefore too low, and a PBR computed from that frame alone is too high:
  **1.5 to 1.8 times** the PBR that uses the background of the exposure-768 frame.
- These numbers are for the two-foci frames of pattern (a) behind the rotating polarizer, on the Fourier-plane camera
  (serial 40238131, Mono12). The paper's single-focus frames are on other frames and partly on the other camera;
  Part B measured the image-plane camera with single foci: see `pbr_bias_check.md`.

## a. The 2 deg scans of pattern (a), exposures 24, 192 and 768

Source frames: `results_small/polarizer_followup_20261115_1342/fine_scans_pattern_a.npz` (recorded 13:44 to 13:47; at
each analyzer angle the three exposures were taken one after the other, with one discarded warm-up frame before each).
Script: `tools/dark_level_analysis.py`. All numbers below are in `results_small/dark_level_analysis.json`
(keys `series`, `summary`, `single_pixel_response_exp24_vs_exp768`, `linear_fit_above_8_counts`).

Definitions used:

- Background: pixels inside the NA disk (radius 58.09 px around (63.5, 63.5) of the 128 x 128 frame) outside the two
  3 x 3 windows of the foci (10594 pixels). Focus positions: A at (47, 56), B at (80, 71). Saturated pixels are left out.
- "Scaled to exposure 24": mean count times 24 / exposure.
- PBR: peak pixel of the transmitted focus over the mean background.
  (i) peak and background from the exposure-24 frame.
  (ii) peak from the exposure-24 frame, background from the exposure-768 frame scaled by 24 / 768.
- Implied offset: background (ii) minus background (i), in counts per pixel at exposure 24.
- Transmitted focus: B in the range -10 to 20 deg (A is near extinction), A in the range 80 to 110 deg.

### At the analyzer angles 6 deg and 96 deg (upward scan / downward scan)

| Quantity | Analyzer 6 deg, focus B transmitted | Analyzer 96 deg, focus A transmitted |
|---|---|---|
| Peak pixel at exposure 24 (counts) | 104 / 104 | 297 / 301 |
| Mean background, exposure 24 (counts) | 0.204 / 0.204 | 0.756 / 0.743 |
| Mean background, exposure 192, scaled to 24 | 0.298 / 0.295 | 1.049 / 1.034 |
| Mean background, exposure 768, scaled to 24 | 0.352 / 0.351 | 1.137 / 1.129 |
| Background pixels reading 0, exposure 24 | 0.906 / 0.907 | 0.772 / 0.776 |
| Background pixels reading 0, exposure 192 | 0.577 / 0.572 | 0.265 / 0.270 |
| Background pixels reading 0, exposure 768 | 0.223 / 0.226 | 0.066 / 0.067 |
| PBR (i), exposure 24 alone | 509 / 509 | 393 / 405 |
| PBR (ii), background from exposure 768 | 295 / 296 | 261 / 267 |
| Ratio (i) / (ii) | 1.72 / 1.72 | 1.51 / 1.52 |
| Implied offset (counts per pixel) | 0.148 / 0.147 | 0.382 / 0.386 |

### Over each scan range (16 angles, upward and downward: 32 frames per range)

| Quantity | -10 to 20 deg, B transmitted | 80 to 110 deg, A transmitted | -10 to 20 deg at stage + 180 deg, B transmitted |
|---|---|---|---|
| Mean background, exposure 24 (counts) | 0.211 | 0.734 | 0.200 |
| Mean background, exposure 192, scaled to 24 | 0.316 | 1.025 | 0.304 |
| Mean background, exposure 768, scaled to 24 | 0.370 | 1.114 | 0.359 |
| Background pixels reading 0 (24 / 192 / 768) | 0.902 / 0.555 / 0.216 | 0.777 / 0.270 / 0.068 | 0.903 / 0.555 / 0.215 |
| Ratio (i) / (ii), mean and spread | 1.75 +- 0.05 | 1.52 +- 0.02 | 1.79 +- 0.05 |
| Ratio (i) / (ii), smallest to largest | 1.65 to 1.87 | 1.49 to 1.56 | 1.69 to 1.89 |
| Implied offset (counts per pixel) | 0.158 +- 0.011 | 0.381 +- 0.011 | 0.159 +- 0.012 |

The spread is the standard deviation over the 32 frames.

With 7 x 7 windows excluded instead of 3 x 3 (10514 background pixels) the ratio (i) / (ii) is 2.11 +- 0.09,
1.62 +- 0.02 and 2.09 +- 0.08 for the three ranges; the implied offset is unchanged within 0.01 counts.

### Response of single pixels: exposure 24 against exposure 768

Background pixels (7 x 7 windows excluded) of all 96 frames, grouped by the count expected from the exposure-768 frame
(count at 768 times 24 / 768):

| Expected at exposure 24 (counts) | Pixels | Mean expected | Mean read at exposure 24 | Read minus expected | Pixels reading 0 |
|---|---|---|---|---|---|
| 0 to 0.125 | 442855 | 0.035 | 0.029 | -0.006 | 0.971 |
| 0.25 to 0.5 | 158393 | 0.340 | 0.083 | -0.257 | 0.919 |
| 0.75 to 1 | 41047 | 0.850 | 0.268 | -0.581 | 0.748 |
| 1.5 to 2 | 21439 | 1.710 | 0.821 | -0.888 | 0.351 |
| 3 to 4 | 10434 | 3.442 | 2.326 | -1.116 | 0.021 |
| 6 to 8 | 3974 | 6.862 | 5.595 | -1.267 | 0.000 |
| 12 to 16 | 1959 | 13.801 | 12.481 | -1.320 | 0.000 |
| 24 to 32 | 338 | 27.693 | 26.024 | -1.669 | 0.000 |

(The JSON has all 16 groups.) Straight-line fit for pixels expected above 8 counts (7349 pixels):
read = 0.988 x expected - 1.19 counts.

### What to keep in mind

- The implied offset per background pixel (0.15 to 0.38 counts) is smaller than the 1.2 counts of a bright pixel because
  most background pixels read 0 and cannot go lower.
- The bias depends on the background level: the lower the background, the larger the ratio (i) / (ii).
- PBR (ii) is itself slightly high. If the exposure-768 frame has the same 1.2-count offset, its background is still
  about 8 % low in the range -10 to 20 deg and about 3 % low in the range 80 to 110 deg. This is an inference from
  the exposure-192 values (they fit the same offset), not a separate measurement.
- Each saved pixel is one bilinear sample of the camera frame, rounded to an integer by `cv2.resize` on uint16 data
  (562 x 562 camera pixels to 128 x 128, 4.39 camera pixels per saved pixel). It is not a sum over camera pixels.
- BlackLevel and Gain of the camera are not set by any lab script (see `manuscript_values.md`, item b), so the values
  in force were whatever the camera held. They were not read out during these frames; on 16 Nov both read 0
  (`pbr_bias_check.md`).
- The same effect was seen in the follow-up report as "3 x 3 window powers read 5 to 9 counts low at exposure 24"
  (`reports/polarizer_followup_20261115_1342/followup_report.md`, section "Low-signal response").

## b. Raw frames of the single-focus measurements

**Raw frames with exposure metadata: not found.** (This is about the frames saved by the standard pipeline. Part B of
16 Nov recorded new single-focus frames with exposure metadata: `pbr_bias_check.md`.) What exists:

- The standard pipeline saves each single-focus frame only as a PNG, `TM_lib/tm_focus_outputs/im_<plane>_<i>.png`
  (`run_tm_focus_optimization.py:1560-1561`). The PNG is peak-normalized: value = round(counts x 4095 / peak) x 16
  (`to_png_uint16`, `run_tm_focus_optimization.py:854-866`). The peak count and the exposure are not stored in the PNG
  or in `run_summary.json`.
- The exposure is hard-coded in the script: 24 in the lab version now on this PC (`run_tm_focus_optimization.py:1535`),
  20 in the older repository version (`TM_acquisition/lab/run_wavefront_control.py:965`). Which version recorded the
  frames used in the paper is not recorded on this PC.
- The frame is the 128 x 128 crop (bilinear resize of 798 x 798 camera pixels in the image plane, 562 x 562 in the
  Fourier plane), flipped in the image plane (`run_tm_focus_optimization.py:1537-1539`).
- Because the counts are integers, they can be recovered from the PNG (script `tools/single_focus_png_recovery.py`,
  output `results_small/single_focus_png_recovery.json`). The recomputed PBR of the two frames of 15 Nov equals the
  value in `run_summary.json` (417.9 and 619.8), which confirms the recovery.

| File (in `TM_lib/tm_focus_outputs`) | Written | Recovered peak (counts) | Mean background (counts) | Background pixels reading 0 | PBR recomputed |
|---|---|---|---|---|---|
| im_image_1.png | 15 Nov 2026 10:55 | 612 | 1.46 | 0.50 | 417.9 |
| im_image_2.png | 2 Nov 2026 11:37 | 1379 | 2.17 | 0.48 | 634.8 |
| im_image_3.png | 2 Nov 2026 11:37 | 1004 | 2.05 | 0.42 | 488.7 |
| im_image_4.png | 2 Nov 2026 11:37 | 1074 | 2.29 | 0.41 | 469.0 |
| im_image_5.png | 2 Nov 2026 11:37 | 1285 | 2.02 | 0.48 | 637.8 |
| im_fourier_1.png | 15 Nov 2026 10:55 | 762 | 1.23 | 0.63 | 619.8 |
| im_fourier_2.png | 2 Nov 2026 11:37 | 799 | 1.38 | 0.56 | 578.4 |
| im_fourier_3.png | 2 Nov 2026 11:37 | 831 | 1.32 | 0.59 | 631.5 |
| im_fourier_4.png | 2 Nov 2026 11:38 | 738 | 1.52 | 0.53 | 484.5 |
| im_fourier_5.png | 2 Nov 2026 11:38 | 1227 | 1.62 | 0.60 | 759.9 |

Notes on this table:

- Frames 2 to 5 are from a run of 2 Nov whose `run_summary.json` and dataset have since been overwritten; their
  target pixels are not recorded, so the peak is the brightest pixel inside the fiber support.
- These files are not copied into this folder (they are older than 12 Nov or belong to the standard pipeline run).

Other saved frames of the standard pipeline, none with exposure metadata:

- `measured_transmission_image_*.npy` and `measured_transmission_fourier_*.npy` in `TM_lib/tm_focus_outputs`
  (float32, 128 x 128, counts) exist for the one-hot-mode, multispot, target-correlation and
  polarization-focus experiments, not for the single-focus frames. `measured_transmission_image.npy` (29 Apr 2026)
  is an 8-bit frame from an older version.
- Benchmark replays: `tm_motor_pcc_outputs/lab_replay/` in the project root (23 Jul 2026) and
  `TM_lib/raf_offline_outputs/lab_replay/` (9 and 10 Nov 2026) hold peak-normalized PNGs and summary tables only
  (`replay_slm_focus_masks.py:415`, `replay_raf_slm_focus_masks.py:385`).
