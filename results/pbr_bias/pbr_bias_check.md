# PBR bias check (Part B)

Recorded 2026-11-16 12:21:03 to 2026-11-16 12:21:41 with `code/run_pbr_bias_check.py`: image-plane camera 40237702 and SLM; the rotation stage was not used.
Run folder: `TM_lib/tm_focus_outputs/pbr_bias_check_20261116_1220`, copied to `results_small/pbr_bias_check_20261116_1220`. Sources: `pbr_bias_check_summary.json` [S] and `pbr_bias_hdr_analysis.json` [H] in that folder; keys are given in brackets.

## Result

- **A PBR taken from one frame at exposure 24 is about 1.4 times too high.** PBR (i) is 1.44 +- 0.04 times the PBR with
  the same background definition (fiber core outside the 3 x 3 window) and a background taken from the long exposures
  (five targets, range 1.37 to 1.48).
- In numbers: PBR (i) is 533 +- 47; with the unbiased background it is 372 +- 43.
- The cause is the dark level. The background of an exposure-24 frame reads 1.67 counts where the long exposures give
  2.38 counts: 0.72 +- 0.04 counts per pixel are missing. A pixel reads about 1.1 counts below a linear response and is
  clipped at 0; half of the background pixels read 0.
- **The ratio (i) / (ii) as specified, 1.05 +- 0.08, does not show this bias.** (ii) leaves out the 7 x 7 window and with
  it the halo of the focus, which holds 34 to 43 % of the background counts of an exposure-24 frame. That raises (ii) by
  about as much as the dark level raises (i). With the 7 x 7 window left out on both sides the ratio is 1.69 +- 0.04.
- The conditions match the saved single-focus frames of the standard pipeline: here the background reads 1.4 to 2.1
  counts with 48 to 54 % of its pixels at 0; there 1.2 to 2.3 counts with 41 to 63 % (`dark_level.md`, section b).
- Camera during these frames: Basler daA1920-160um, Mono12, BlackLevel 0, Gain 0 dB, exposure 24 microseconds.
- A first attempt at 11:38 was invalid because the SLM did not modulate the light; it is described at the end.

## Things to know

- **How well the factor is known.** The spread over the five targets is 0.04. On top of that the scaling between
  exposures is only good to about 3 %: per unit exposure the camera gives about 1 % more signal for every factor of 4 in
  exposure (table "Exposure linearity"). With the background taken from the exposures up to 384 the factor is 1.40
  instead of 1.44. A fair statement is 1.4, within 0.05.
- **The factor depends on the background level.** In this run it is 1.37 for the brightest background (2.09 counts
  read) and 1.48 for the dimmest (1.40 counts read). It should not be used for frames whose background is very different.
- **What it gives for the saved image-plane frames.** Dividing the PBR of `im_image_1.png` to `im_image_5.png`
  (417.9, 634.8, 488.7, 469.0, 637.8; `dark_level.md`) by 1.44 gives about 290, 441, 339, 326 and 443. This assumes the
  camera settings on 2 Nov and 15 Nov were those read today; no script sets BlackLevel or Gain, and they were read out
  for the first time in this run.
- **Fourier-plane camera.** Not measured with single foci (Part B was for the image plane only). It is the same model
  with the same settings, and in the two-foci frames its pixels read 1.2 counts low at exposure 24 (`dark_level.md`,
  section a), so a similar factor is to be expected there. That is an inference, not a measurement.
- **Target 5.** Its five frames at exposure 384 are not alike: the pattern of the weak light around the focus changes
  from frame to frame (correlation with the median frame 0.89 to 1.00; 0.998 or more for the other targets), while
  the total counts agree within 1 %. The cause is not known. Its PBR values rest on the exposure-24 and exposure-1536
  frames, which are consistent, and its factor (1.475) is in line with the others; only its linearity fits are left out.
- **The TM was 25.6 hours old** and still reproduced the dataset frames at 0.947. The fiber had not drifted; the
  freshness of 0.021 in the first attempt was the SLM.

## Camera state during the frames

| Setting | Image-plane camera (used) | Fourier-plane camera (settings only) |
|---|---|---|
| Serial | 40237702 | 40238131 |
| Model | daA1920-160um | daA1920-160um |
| Firmware string | p=du1_imx392m/s=r/v=1.2.1/i=6515.14/h=61c0643 | p=du1_imx392m/s=r/v=1.2.1/i=6515.14/h=61c0643 |
| Pixel format | Mono12 | Mono12 |
| BlackLevel | 0.0 DN (range 0.0 to 4095.0) | 0.0 DN (range 0.0 to 4095.0) |
| Gain | 0.0 dB (range 0 to 48) | 0.0 dB (range 0 to 48) |
| GainAuto / ExposureAuto | Off / Off | Off / Off |
| Gamma / DigitalShift / contrast mode | 1.0 / 0 / Linear | 1.0 / 0 / Linear |
| Binning (horizontal, vertical) | 1, 1 | 1, 1 |
| Frame (width x height), offset | 1920 x 1200, (8, 8) | 1920 x 1200, (8, 8) |
| Exposure unit; minimum in this format | us; 24.0 | us; 24.0 |
| USB speed mode | SuperSpeed | SuperSpeed |

- Read from the cameras before the frames [S `camera_state_before`] and after them [S `camera_state_after`]: pixel format, BlackLevel, Gain, GainAuto and ExposureAuto are the same at both times.
- The script reads these settings and changes none of them. Pixel format and exposure are set by the standard `setup()` of the pipeline.
- E0 = 24, the exposure of the single-focus capture in the lab script and also the camera minimum in Mono12. Exposures recorded: 24, 96, 384, 1536 [S `exposures`]. Exposure 20 cannot be set in Mono12 on this camera.
- After the run the pipeline's `teardown()` put both transmission cameras back to Mono8 at the minimum exposure (19) and released the SLM, as after every standard run (last two lines of `log.txt`).

## Freshness of the TM

| Check | Correlation with the dataset frames, 8 masks | Between frames of different masks | Level now / dataset | Minutes after the dataset |
|---|---|---|---|---|
| Start | 0.947 | 0.005 | 1.00 | 1538 |
| End | 0.943 | 0.005 | 1.00 | 1538 |

- TM used: standard checkpoint (`tm_system_checkpoint.pt` of 15 Nov 10:54, dataset of 15 Nov 10:43). No new dataset was needed in this run.
- "Between frames of different masks" is the guard added after the first attempt: it is near 0 when the SLM modulates the light.

## Per target, as specified

PBR = peak pixel / mean of the fiber core (radius 58.12 px) outside the 3 x 3 window, on the 128 x 128 frame as the pipeline evaluates it (crop, bilinear resize, flip). Five frames per exposure.

| Target (y, x) | Peak at E0 (counts) | PBR (i) | PBR (ii) | PBR (iii) | (i) / (ii) | (i) / (iii) | Background at E0 (counts) | Background pixels at 0, E0 |
|---|---|---|---|---|---|---|---|---|
| 1: (68, 66) | 835 | 488 +- 7 | 465 | 481 | 1.049 | 1.014 | 1.71 | 0.496 |
| 2: (59, 67) | 802 | 526 +- 8 | 497 | 516 | 1.060 | 1.020 | 1.53 | 0.538 |
| 3: (60, 69) | 1283 | 613 +- 13 | 657 | 678 | 0.933 | 0.904 | 2.09 | 0.475 |
| 4: (66, 68) | 846 | 528 +- 8 | 482 | 499 | 1.096 | 1.058 | 1.60 | 0.485 |
| 5: (58, 57) | 715 | 512 +- 8 | 452 | 468 | 1.132 | 1.094 | 1.40 | 0.528 |
| Mean +- spread (range) | | 533 +- 47 (488 to 613) | 510 +- 83 (452 to 657) | 528 +- 86 (468 to 678) | 1.054 +- 0.075 (0.933 to 1.132) | 1.018 +- 0.071 (0.904 to 1.094) | 1.67 +- 0.26 (1.40 to 2.09) | 0.504 +- 0.027 (0.475 to 0.538) |

- (i): one frame at E0, as in the paper; the value is the mean of the five frames and +- is their standard deviation.
- (ii): peak from E0, background from exposure 1536 (the longest; 0.02 % of its background pixels read 0, so the 1 % rule is met for all targets), without saturated pixels and without a 7 x 7 window around the peak, scaled by the exposure ratio.
- (iii): the same with exposure 384.
- Last row: mean and standard deviation over the five targets, range in brackets [S `summary`].

## The same comparison with one background definition

(i) leaves out a 3 x 3 window and (ii) a 7 x 7 window. The 40 pixels between the two windows are the halo of the focus; they are bright and belong to the background of (i) but not of (ii). To compare like with like, every pixel was given the value of the longest exposure at which it is not saturated, scaled to E0 (script `tools/pbr_bias_hdr_analysis.py`). The peak is always the E0 peak.

| Target | PBR (i) | PBR, same 3 x 3 definition, background from the long exposures | (i) / that | Background, E0 -> long exposures (counts) | Missing per background pixel (counts) | Halo ring: mean count, share of the E0 background | Both with 7 x 7 excluded: ratio |
|---|---|---|---|---|---|---|---|
| 1 | 488 | 344 | 1.419 | 1.71 -> 2.43 | 0.72 | 165, 36 % | 1.648 |
| 2 | 526 | 362 | 1.454 | 1.53 -> 2.22 | 0.69 | 158, 39 % | 1.736 |
| 3 | 613 | 448 | 1.367 | 2.09 -> 2.86 | 0.77 | 239, 43 % | 1.638 |
| 4 | 528 | 362 | 1.461 | 1.60 -> 2.34 | 0.74 | 154, 36 % | 1.712 |
| 5 | 512 | 347 | 1.475 | 1.40 -> 2.06 | 0.66 | 126, 34 % | 1.708 |
| Mean +- spread (range) | 533 +- 47 (488 to 613) | 372 +- 43 (344 to 448) | 1.435 +- 0.043 (1.367 to 1.475) | 1.67 -> 2.38 | 0.72 +- 0.04 (0.66 to 0.77) | | 1.688 +- 0.043 (1.638 to 1.736) |

- Where the E0 background goes missing, target 1 [H `targets[0].zones`]: the 40 ring pixels read 6614 counts at E0 against 6736 from the longer exposures; the other 10562 background pixels read 11494 against 18975 (mean 1.09 against 1.80 counts; 50 % of them read 0 at E0).
- Source exposure of the background pixels, target 1: 13 pixels from exposure 96, 19 pixels from exposure 384, 10571 pixels from exposure 1536.
- Leaving out saturated pixels hardly matters here: outside the 7 x 7 window 0 to 1 pixels per target saturate at the longest exposure; (ii) as specified equals the composite value with 7 x 7 excluded within 0.3 % [H `ratio_ii_as_specified_over_hdr_7x7`].
- With the composite built from the exposures up to 384 only, the ratio of the fourth column is 1.399 +- 0.042 (1.335 to 1.443) [H `ratio_i_over_hdr_3x3_up_to_384`].
- Check of the scaling on the peak itself: peak from exposure 96 scaled to E0 / peak at E0 = 1.005 +- 0.005 (1.000 to 1.010) (targets whose peak is not saturated at 96).

## Background against exposure (mean over the targets, outside the 7 x 7 window)

| Exposure | Mean background (counts) | Scaled to E0 | Background pixels at 0 | Saturated camera pixels | Peak pixel (counts) |
|---|---|---|---|---|---|
| 24 | 1.03 | 1.034 | 0.5062 | 0 | 896 |
| 96 | 5.93 | 1.483 | 0.0997 | 34 | 3387 |
| 384 | 26.91 | 1.682 | 0.0074 | 795 | saturated |
| 1536 | 111.40 | 1.741 | 0.0001 | 1438 | saturated |

## Response of single pixels at E0

Background pixels of the 128 x 128 frames outside the 7 x 7 window, all targets and all five E0 frames, grouped by the count expected from the longer exposures (scaled to E0) [H `single_pixel_response_e0`]:

| Expected at E0 (counts) | Pixel readings | Mean expected | Mean read at E0 | Read minus expected | Readings of 0 |
|---|---|---|---|---|---|
| 0 to 0.25 | 13155 | 0.160 | 0.047 | -0.113 | 0.954 |
| 0.25 to 0.5 | 28430 | 0.378 | 0.102 | -0.276 | 0.902 |
| 0.5 to 0.75 | 31845 | 0.623 | 0.174 | -0.449 | 0.835 |
| 0.75 to 1 | 31225 | 0.872 | 0.286 | -0.585 | 0.734 |
| 1 to 1.5 | 51435 | 1.235 | 0.499 | -0.736 | 0.562 |
| 1.5 to 2 | 36060 | 1.732 | 0.866 | -0.865 | 0.329 |
| 2 to 3 | 38110 | 2.422 | 1.434 | -0.987 | 0.128 |
| 3 to 4 | 16425 | 3.425 | 2.346 | -1.079 | 0.022 |
| 4 to 6 | 10460 | 4.754 | 3.623 | -1.131 | 0.002 |
| 6 to 8 | 2890 | 6.891 | 5.678 | -1.213 | 0.000 |
| 8 to 12 | 2165 | 9.673 | 8.439 | -1.234 | 0.000 |
| 12 to 16 | 845 | 13.856 | 12.452 | -1.404 | 0.000 |
| 16 to 24 | 710 | 19.085 | 17.654 | -1.431 | 0.000 |
| 24 to 32 | 235 | 27.348 | 25.685 | -1.663 | 0.000 |
| 32 to 64 | 85 | 41.516 | 39.859 | -1.657 | 0.000 |

Straight-line fit between 8 and 512 expected counts (4040 readings): read = 0.985 x expected -1.13 counts [H `linear_fit_8_to_512_counts`].

## Exposure linearity (camera pixels between 50 and 3000 counts at both exposures)

| Exposures | Pixels per target | Fit slope | Fit offset at the longer exposure (counts) | Implied dark offset (counts) |
|---|---|---|---|---|
| 24 and 96 | 1303 to 1475 | 1.0070 +- 0.0007 | 3.2 +- 0.7 | -1.07 +- 0.24 |
| 24 and 384 | 556 to 640 | 1.0171 +- 0.0026 | 16.0 +- 3.6 | -1.07 +- 0.24 |
| 24 and 1536 | 0 to 0 | too few pixels | | |
| 96 and 384 | 2910 to 4664 | 1.0085 +- 0.0017 | 4.0 +- 0.8 | -1.33 +- 0.28 |
| 96 and 1536 | 2256 to 4065 | 1.0271 +- 0.0043 | 7.7 +- 3.7 | -0.52 +- 0.25 |
| 384 and 1536 | 37668 to 54569 | 1.0112 +- 0.0030 | 1.4 +- 0.9 | -0.47 +- 0.30 |

- Raw camera crops (798 x 798 pixels), mean of the five frames at each exposure, pixels unsaturated in every frame. Fit: count at the longer exposure = slope x (count at the shorter x exposure ratio) + offset. Mean +- spread over the targets 1, 2, 3, 4.
- Implied dark offset = - offset / (exposure ratio - 1): the constant that would have to be added to every reading for the two exposures to be proportional.
- Target 5 is left out of this table only: see "Things to know".

## First attempt, 11:38 to 12:06: invalid

Run folder `TM_lib/tm_focus_outputs/pbr_bias_check_20261116_1139` (summary, log and note copied to `results_small/pbr_bias_check_20261116_1139`).

- The SLM did not modulate the light: every mask gave the same speckle. Freshness 0.021; light level at the image camera 2.84 times the dataset level; nothing outside the fiber core.
- Taking the low freshness for drift, the script recorded a new dataset (14.8 min) and fitted a new TM (11.5 min). Both are useless; they were renamed with the suffix `_INVALID_slm_not_modulating` and then, at Dvir's request, moved to the Windows Recycle Bin. No focus formed (peak 139 to 229 counts on a background of 29 counts).
- **None of the PBR numbers of that run are a result.** Its camera settings are the same as in the table above.
- From the PC everything looked normal (Windows not locked, both displays active, Blink SDK constructed). The cause was not found from the PC. The second run was started at 12:20 after Dvir wrote that he hoped it was fixed, and it worked; what had been wrong is not recorded.
- The script now stops after the first 8 probe frames if frames of different masks are correlated above 0.5.

## Other facts of the run

- Targets were drawn with `sample_target_pixel_near_center` (numpy seed 20261115) and optimized with `optimize_slm_pattern_with_tm` (150 iterations, learning rate 0.1, 1 x 1 window), as in the standard run. PBR predicted by the TM: 588, 583, 650, 616, 603 [S `targets[].tm_predicted_pbr`].
- One warm-up frame was discarded after every change of pattern or exposure: 28 up to the last focus frame, 36 with the freshness check at the end (`log.txt`).
- Saved frames: `focus_frames_128.npz` (128 x 128, as evaluated, with the exposures) and `target_<k>_raw_crops_uint16.npz` (798 x 798 camera pixels, not resized, not flipped, five frames per exposure). These are raw single-focus frames with exposure metadata.
- SLM patterns: `target_<k>_phase_uint8.npy` and `target_<k>_phase_rad.npy`.
