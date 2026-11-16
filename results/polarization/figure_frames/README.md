# Frames for the paper figure

Made by `tools/make_figure_frames.py` on 15 Nov 2026. Values are copied out of the saved result files; nothing is
rescaled. Per-frame metadata (source file and index, stage angle, exposure, saturation flags): `figure_frames.json`.

## What a frame is

- Camera: transmission Fourier-plane camera, serial 40238131, pixel format Mono12 (counts 0 to 4095).
  BlackLevel and Gain: not set by the scripts and not read out (see `manuscript_values.md`, item b).
- Exposure: the number passed to `ExposureTime` (microseconds).
- Geometry: the full sensor frame was cropped to 562 x 562 camera pixels around the centre of the image, which the
  wedge of the rotating polarizer displaces (circle of radius 9.8 px on the 128 grid), and resized to 128 x 128 with
  `cv2.resize(..., INTER_LINEAR)` on 16-bit data. Each saved pixel is one bilinear sample of the camera counts, rounded
  to an integer; it is not a sum over the 4.39 x 4.39 camera pixels it stands for.
- **The full sensor frames were never saved.** These 128 x 128 frames are the rawest data that exist.
- No dark subtraction, no normalization.
- Each frame is stored twice: `<name>.npy` (float32, counts) and `<name>_counts.png` (16-bit PNG whose pixel value is
  the count itself, so it looks black in an image viewer).

## Pattern (a) at stage 6 deg and 96 deg

Pattern (a) is the x'/y' two-foci pattern of the weighted TM, designed with no polarizer information
(SLM pattern: `results_small/polarizer_test_20261115_1201/two_foci_probe_xy_weighted_tm_phase_uint8.npy`).
Focus A is at (48, 56), focus B at (80, 72) (y, x). At stage 6 deg focus B passes; at stage 96 deg focus A passes.

| Files | Stage angle | Exposures | Recorded | Source |
|---|---|---|---|---|
| `pattern_a_analyzer_006deg_stage_006deg_exposure_{0024,0192,0768}_scan_{up,down}` | 6 deg | 24, 192, 768 | 15 Nov 13:44 to 13:47 | 2 deg scans, `fine_scans_pattern_a.npz` |
| `pattern_a_analyzer_096deg_stage_096deg_exposure_{0024,0192,0768}_scan_{up,down}` | 96 deg | 24, 192, 768 | same | same |
| `pattern_a_analyzer_006deg_stage_186deg_exposure_{0024,0192,0768}_scan_{up,down}` | 186 deg (the same analyzer as 6 deg) | 24, 192, 768 | same | same |
| `pattern_a_analyzer_{006,096}deg_stage_{006,096}deg_exposure_0024_test_a_record` | 6 and 96 deg | 24 | 15 Nov 13:49 | `two_foci_x6_probe_xy_weighted_tm_frames.npz`; the report's -0.976 comes from this pair |

- At exposure 768 and stage 96 deg the peak of focus A is saturated (4095 counts); `figure_frames.json` flags it
  (`saturated_near_A`). No other frame of this list is saturated at a focus.
- At stage 186 deg focus B is about 20 % weaker than at stage 6 deg (see the follow-up report, "Full scan").
- "scan_up" and "scan_down" are the same angle reached from below and from above.
- At exposure 24 the camera reads low signals about 1.2 counts too low and clips at 0 (`dark_level.md`).
- In `figure_frames.json`, `window_3x3_counts` is the sum around the brightest pixel within 3 px of the target. For a
  blocked focus that pixel is a noise pixel, so use these numbers only for the transmitted focus.

**A frame of pattern (a) without polarizer: not found.** None was recorded.

## The three images of `best_two_foci_three_images`

`image_polarizer_along_x_stage_6deg.png`, `image_polarizer_along_y_stage_96deg.png` and `image_no_polarizer.png`
show **pattern (b), not pattern (a)**:

| Question | Answer | Source |
|---|---|---|
| Pattern | lab-axes pattern of the weighted TM with U from the foci calibration at stage 6 / 96 deg, tag `x6_lab_axes_U_from_foci_v2`: focus A along lab x (stage 6 deg), focus B along lab y (stage 96 deg) | `info.json`, keys `pattern`, `what` |
| SLM pattern file | `results_small/polarizer_followup_20261115_1342/two_foci_x6_lab_axes_U_from_foci_v2_phase_uint8.npy` (copy: `results_small/best_two_foci_three_images/slm_pattern_phase_uint8.npy`) | |
| Exposure | 24 for all three, Mono12, Fourier camera | `info.json`, keys `with_polarizer.exposure`, `without_polarizer.exposure` |
| Recorded, with polarizer | 15 Nov 2026 14:05:43 (202 min after the dataset) | `info.json`, key `with_polarizer.recorded` |
| Recorded, without polarizer | 15 Nov 2026 15:12:15 (269 min after the dataset; both polarizers out of the beam) | `info.json`, key `without_polarizer.recorded` |
| Scaling of the three PNGs | 16-bit, linear, one common scale: value = round(counts x 65535 / 242); 242 counts is the largest count of the three frames | `code/collect_best_two_foci_images.py`, function `write_images`; `info.json`, key `png_full_scale_counts` |
| Unscaled data | the `.npy` files next to the PNGs hold the counts | |

The same three frames are in this folder as `pattern_b_v2_image_*.npy` and `*_counts.png` (counts, unscaled).

- The two polarizer images are 67 minutes older than the image without polarizer.
- Without polarizer the image is not displaced; the frame was cropped around the nominal centre.
- The other PNGs of the result folders named `im_two_foci_<tag>_pol_0deg.png` and `..._pol_90deg.png` use the same
  kind of scaling, with one scale per pair (`run_polarization_axes.py`, function `_save_png16_common`).
