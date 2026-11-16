# Follow-up of the polarizer test, 15 Nov 2026, 13:42 to 14:06

Weighted TM of 15 Nov (b = +0.701), dataset of 10:43. Pattern (a) is the x'/y' pattern of the weighted TM from the first run
(focus A in x', focus B in y'). Angles are on the stage scale. The polarization direction of a focus behind the arm is
its extinction angle + 90 deg. "Before the arm" divides the field components along the arm axes by sqrt(1 + d) and sqrt(1 - d).

## Main results

1. **Arm axes: stage 4.9 deg and 94.9 deg.** The pair of foci whose two directions stay 90 deg apart behind the arm lies at
   phi = +3.6 deg, with the directions at stage 4.9 and 94.9 deg. The arm axes are neither at stage 0 nor exactly at stage 6 deg.
2. **Pattern (a):** focus A extinguishes at +4.53 +- 0.24 deg and focus B at +99.07 +- 0.68 deg. Behind the arm the two
   directions are 94.5 +- 0.7 deg apart.
3. **Before the arm (d = -0.70):** 83.2 +- 0.6 deg apart for arm axes at stage 0, and 94.8 +- 0.6 deg for arm axes at stage 6 deg.
   Neither is 90 deg. For the measured arm axes (stage 4.9 deg) the angle is 92.6 +- 0.6 deg.
4. **Lab x at stage 6 deg:** test (a) gives -0.976 raw and test (b) gives +0.990 raw (+0.998 before the arm), at exposure 24.
   These values are a little optimistic, see "Low-signal response" below.

## Guard against a one-frame lag

- The rig took one warm-up frame, discarded, after every change of stage angle, SLM pattern or exposure: 1450 warm-up frames in the main run.
- Lag indicator: (I_0 + I_90) against the dataset frame of the previous mask is 0.04; against the same mask 0.951.
- Upward and downward scans agree in every case: at most 0.12 deg for pattern (a) and 0.31 deg for the pairs (limit 0.5 deg).
- The same range at stage angle + 180 deg gives the same extinction angle of focus A (+4.51 deg against +4.53 deg).
- A dry run without hardware with a simulated one-frame lag gave unchanged results.

## Pattern (a): 2 deg scans (three exposures: 24, 192, 768; the focus near extinction is evaluated at 768)

| Quantity | Focus A | Focus B |
|---|---|---|
| Extinction angle, upward | +4.50 deg | +99.13 deg |
| Extinction angle, downward | +4.56 deg | +99.01 deg |
| Mean and uncertainty | +4.53 +- 0.24 deg | +99.07 +- 0.68 deg |
| Statistical part (fit, bootstrap, up/down) | 0.06 deg | 0.18 deg |
| Spread between 3x3 and 5x5 windows and the exposures | 0.46 deg | 1.31 deg |
| Minimum / maximum of the fitted power | 0.0004 | 0.036 |
| Polarization direction behind the arm | 94.53 +- 0.24 deg | 9.07 +- 0.68 deg |

The uncertainty is the statistical part combined with half of the spread. Focus B does not extinguish fully (3.6 %), so it is
not purely linear.

Angle between the two directions behind the arm: **94.5 +- 0.7 deg** (85.5 deg measured the other way).

| Arm axes at | Direction of A before the arm | Direction of B before the arm | Angle between | Deviation from 90 deg |
|---|---|---|---|---|
| stage 0 | 100.7 +- 0.6 deg | 3.8 +- 0.3 deg | 83.2 +- 0.6 deg | -6.8 deg |
| stage 6 deg | 92.5 +- 0.6 deg | 7.3 +- 0.3 deg | 94.8 +- 0.6 deg | +4.8 deg |
| stage 4.9 deg (measured with the pairs) | 94.1 deg | 6.6 deg | 92.6 +- 0.6 deg | +2.6 deg |

(d = -0.70.) The two directions would be exactly 90 deg apart for arm axes at stage 3.6 deg.

Pass fractions of pattern (a) read from the fine scan (mean of upward and downward):

| Analyzer pair | Exposure | A passes | B passes | Separation |
|---|---|---|---|---|
| 0 / 90 deg | 24 | 0.0026 | 0.9489 | -0.946 |
| 0 / 90 deg | longest unsaturated for each focus | 0.0079 | 0.9445 | -0.937 |
| 6 / 96 deg | 24 | 0.0003 | 0.9788 | -0.979 |
| 6 / 96 deg | longest unsaturated for each focus | 0.0008 | 0.9686 | -0.968 |

## Full scan of pattern (a), 0 to 175 deg in steps of 5 deg (exposure 24, no saturation)

Figure: `full_scan_pattern_a.png`; data: `full_scan_pattern_a.npz`.
The curves are not pure Malus curves (residuals of 7 % and 12 % of the amplitude). The power of focus B at stage angle + 180 deg
is 20 % lower than at the stage angle itself (27 % at -10 deg, 15 % at +20 deg), for 3x3, 5x5 and 9x9 windows alike, while focus A
agrees within 5 %. This is a throughput that depends on the stage angle and on the focus, not on the analyzer angle; upward and
downward scans agree within a few percent, so it is not a drift. It shifts the azimuths of Malus fits over the full range
(A 90.8 deg, B 11.9 deg) but hardly the extinction angles.

## Lab x at stage 6 deg (stage 6 and 96 deg, one exposure of 24, no saturation)

| | Test (a): x'/y' of the weighted TM | Test (b): U from the foci calibration at 6 / 96 deg |
|---|---|---|
| Where the foci appear | A at lab y, B at lab x | A at lab x, B at lab y |
| Pass fraction at lab x, raw (A, B) | 0.000, 0.976 | 0.990, 0.000 |
| Separation, raw | -0.976 | +0.990 |
| Separation before the arm (d = -0.705) | -0.996 | +0.998 |
| Leak of the blocked focus at lab x / lab y | 0.000 / 0.009 | 0.000 / 0.005 |
| PBR of focus A at lab x / lab y | 15 / 398 | 603 / 6 |
| PBR of focus B at lab x / lab y | 527 / 7 | 11 / 493 |
| Scan of (b): azimuth from lab x, degree of linear polarization | | A 9.7 deg, 1.05; B 89.2 deg, 1.07 |

Foci calibration at 6 / 96 deg: d = -0.705 (from the +45'/-45' and R'/L' pairs; per basis -0.914, -0.736, -0.675);
R1 = (-0.997, -0.016, -0.070), 4.1 deg from (-1, 0, 0) on the Poincare sphere (15.8 deg with lab x at stage 0);
norm of R1 0.998; pair imbalance -0.008, -0.128, 0.117. All three checks pass.

## Low-signal response of the camera (affects every result taken at exposure 24)

At exposure 24 the 3x3 window powers read 5 to 9 counts lower than the same powers at exposure 768 (scaled): the dark level
is clipped at zero. A blocked focus therefore reads too low; in tests (a) and (b) above it reads exactly 0 counts.
Estimates with this offset added: test (a) about -0.96 raw; test (b) about +0.98 raw and +0.97 before the arm.
For the first run (lab x at stage 0): (a) about -0.95 raw and -0.96 before the arm; (b) about +0.94 raw and +0.96 before the arm.
The verdict of the first run (above 0.9) does not change. The long-exposure rows of the fine-scan table are not affected.

## Optional: pairs linear before the arm along stage phi and phi + 90 deg

Designed with the speckle frame of the first run (U with its phase, theta0 = 6.0 deg), as the 3f variant "linear before the arm".
Freshness before the pairs: 0.945 (188 min after the dataset).

| phi | Extinction A (up / down) | Extinction B (up / down) | Direction A behind the arm | Direction B behind the arm | Angle between |
|---|---|---|---|---|---|
| -4 deg | 75.39 deg (75.54 / 75.23) | 1.41 deg (1.40 / 1.42) | -14.6 deg | 91.4 deg | 106.0 +- 0.6 deg |
| 0 deg | 85.98 deg (85.94 / 86.03) | 2.99 deg (2.98 / 3.00) | -4.0 deg | 93.0 deg | 97.0 +- 0.4 deg |
| 4 deg | 96.11 deg (96.26 / 95.95) | 5.54 deg (5.55 / 5.53) | 6.1 deg | 95.5 deg | 89.4 +- 0.5 deg |
| 8 deg | 106.46 deg (106.49 / 106.42) | 6.78 deg (6.77 / 6.79) | 16.5 deg | 96.8 deg | 80.3 +- 0.5 deg |
| 12 deg | 115.71 deg (115.74 / 115.68) | 8.56 deg (8.61 / 8.51) | 25.7 deg | 98.6 deg | 72.9 +- 0.5 deg |

- Linear fit of the angle against phi: 90 deg at **phi = +3.58 +- 0.11 deg**; the directions there are stage **4.87 deg and 94.87 deg**.
- Fit of the squeeze of the arm to the ten directions: arm axes at stage 4.90 +- 0.16 deg and 94.90 deg; d = -0.737 +- 0.006;
  the designed direction before the arm lies 1.35 +- 0.11 deg above its nominal value; rms residual 0.39 deg.
  The uncertainties are statistical; the choice of window alone moves an extinction angle by up to 0.5 deg.

## What this says about the axes

- Arm axes (pairs): stage 4.9 and 94.9 deg.
- Axes of the TM: x' at 94.1 deg and y' at 6.6 deg before the arm (pattern (a)); theta0 = 6.0 deg from speckle (first run);
  R1 is 4.1 deg from (-1, 0, 0) on the sphere with lab x at stage 6 deg, which is 2 deg in real space.
- So the axes of the TM coincide with the axes of the arm within about 2 deg.
- d: -0.705 from the foci calibration, -0.706 from speckle, -0.737 from the squeeze of the pairs; learned b = +0.701.

## Problems met

- **NaN in frame_from_focus_pairs.** With lab x at stage 6 deg, focus A of the x'/y' pair read exactly 0 counts at lab x. The pair
  is then exactly pure and the estimator divides 0 by 0. Test (b) of the main run was designed from that invalid matrix; its files
  are kept with the suffix `_INVALID_nan_calibration`. I recomputed the calibration from the same recorded frames with a common d
  (`run_polarizer_followup_b.py`) and re-recorded test (b) at 14:05 (freshness 0.938). `polarizer_arm_estimator.py` is unchanged.
- **Low-signal response** and **throughput against stage angle**: described above.

## Times and freshness

| Event | Time | After the dataset | Freshness (8 masks) |
|---|---|---|---|
| Start, freshness | 13:42 | 179 min | 0.951 |
| 2 deg scans of pattern (a) | 13:44 to 13:47 | 181 to 184 min | |
| Full scan, tests at lab x = 6 deg | 13:48 to 13:51 | 185 to 188 min | |
| Pairs | 13:51 to 14:03 | 188 to 200 min | 0.945 |
| Test (b) re-recorded | 14:04 to 14:05 | 201 to 202 min | 0.938 |

## Code and files

No protected file was changed. New or changed scripts in TM_lib: `run_polarizer_followup.py` (rewritten: frame guard, upward and
downward scans, full scan, pairs), `run_polarizer_followup_b.py` (repair of test (b)), `stageB_followup_analysis.py` (offline).

In this folder: `followup_report.md`, `polarizer_followup_summary.json`, `polarizer_followup_b_summary.json`,
`followup_extra_analysis.json`, logs, `fine_scans_pattern_a.npz`, `full_scan_pattern_a.npz` and `.png`,
`fine_scans_pair_before_arm_phi*.npz`, the SLM patterns `two_foci_pair_before_arm_phi*_{phase_rad, phase_uint8,
full_slm_mask_uint8}.npy` and `_phase_hsv.png`, the frames and images `two_foci_x6_*`, `jones_matrix_U_from_foci_x6_v2.npy`,
and `freshness_frames_*.npz`.
