# Stage B report (revised plan), 15 Nov 2026

Dataset: `measurement_focus.h5` of 10:43 (rotating polarizer out of the beam). Polarizer test: 12:02 to 12:12,
polarizer in front of the Fourier camera (40238131). Lab x = stage 0 deg, lab y = stage 90 deg.
tau = P_0 / (P_0 + P_90) is the fraction of a focus that passes at stage 0; separation = tau_A - tau_B.
"Before the arm" = powers divided by (1 + d, 1 - d) with d = -0.693 from the foci calibration (step 3d).

## Verdict

**Good.** Test (b), designed with U from the foci calibration, has a separation of +0.984 before the arm (+0.962 raw).
Test (a), designed from the weighted TM alone with no polarizer information, has -0.976 before the arm (-0.962 raw):
it reaches 0.9 in magnitude, with the two labels swapped (focus A, in x', appears at stage 90; focus B, in y', at stage 0).

Both parts of the hypothesis hold, with one qualification: the stage zero lies about 6 deg from the axes of the TM.

## Step 1. New dataset and blind TM

| Quantity | 15 Nov | 12 Nov |
|---|---|---|
| Speckle contrast, Fourier plane (mean envelope divided out) | 0.821 | 0.822 |
| Speckle contrast, image plane | 0.745 | 1.033 |
| Mean level in the support, Fourier plane | 427 counts | 427 counts |
| Mean level in the support, image plane | 273 counts | 36 counts |
| Held-out correlation of the blind TM, image plane | 0.913 | 0.792 |
| Held-out correlation of the blind TM, Fourier plane | 0.929 | 0.829 |

The Fourier contrast is not near 1, so the rotating polarizer was out of the beam and the dataset is usable.

## Step 2. b learned by gradient descent (200 epochs, settings of the original run)

| Run | Learned b | Plane scale (fitted afterwards) | Held-out image | Held-out Fourier |
|---|---|---|---|---|
| b from +0.47 | +0.7010 | 1.237 | 0.956 | 0.974 |
| b from -0.47 | -0.7009 | 1.237 | 0.956 | 0.974 |
| b from 0 | +0.7008 | 1.237 | 0.956 | 0.974 |
| Blind, 200 epochs | 0 (fixed) | 1.079 | 0.915 | 0.931 |
| Blind checkpoint (standard run, 100 epochs) | | | 0.913 | 0.929 |

- The three runs give the same abs(b) = 0.701 (within 0.0002), flat from epoch 60 on. No runs with fixed b were needed.
- abs(b) is 0.70, not 0.47. It lies just above the range of 0.45 to 0.68 from the fringes.
- Plane scale: the loss is a Pearson correlation per plane, so it does not change when a plane is rescaled
  (checked numerically). A plane scale cannot be learned by it and cannot bias b. The table gives the
  least-squares scale after the retrieval, relative to the present constant, for weights normalised to (1 + b, 1 - b).
- Axes of independent retrievals (not in the plan): the three learned TMs have the same polarization axes
  (0.0 deg apart on the Poincare sphere, explained fraction 1.000) and differ only by the phase between x' and y'
  (44 and 103 deg). The two blind TMs lie 48 and 80 deg away.
- Kept: `tm_focus_outputs/tm_system_checkpoint_camweights_learned_b_20261115.pt` (b = +0.701; run "b from +0.47").

## Step 3. Polarizer test

### 3a. Displacement

- Circle of radius 9.8 px around (-0.2, +0.1) px; values at all 12 scan angles are in the summary file.
- Cross-correlation with the dataset frames (no model) against the outline of the NA disk in the mean of 104 frames:
  differences 0.24, 0.16, 0.23, 0.10 px at 0/45/90/135 deg (limit 0.5 px).
- Against the TM prediction: at most 0.04 px.
- Every frame was cropped from the raw camera frame around the displaced centre.

### 3b. Freshness

| Quantity | Value |
|---|---|
| (I_0 + I_90) against the dataset frames, first 8 masks, 81 min after the dataset | 0.943 (0.925 to 0.953) |
| The same, all 104 masks | 0.948 |
| The same 8 masks, 89 min after the dataset (during 3f) | 0.945 |
| Level factor (I_0 + I_90) / dataset frame | 0.767 (Stage A: 0.78) |
| Mean light | 77.8 % along stage 90, degree of linear polarization 0.56, azimuth 90.0 deg |

The freshness is above 0.9, so no new dataset was recorded.

### 3c. Axes without calibration (speckle, weighted TM, 104 masks; the first 8 masks give the same within 0.01)

| Quantity | Value |
|---|---|
| abs(U) | [[0.008, 1.000], [1.000, 0.008]]: a swap of the two labels |
| Cross-term power | 0.01 % |
| x' at the polarizer | azimuth 89.5 deg, ellipticity 0.0 deg, in axes turned by theta0 |
| theta0 | +6.0 deg |
| d (three estimates) | -0.706, 0.699, -0.707 |
| Explained fraction | 0.89 |
| Correlation per angle, 0/45/90/135 deg | 0.916, 0.927, 0.932, 0.950 |
| Phase between x' and y' | -42.7 deg (two halves of the masks: -42.6 and -42.7 deg) |

### 3d. Calibration with foci

| Quantity | Value |
|---|---|
| d from frame_from_focus_pairs | -0.693 |
| d per basis (x'/y', +45'/-45', R'/L') | -0.535, -0.742, -0.654 |
| d from the two pairs that constrain it (+45'/-45', R'/L') | -0.697 |
| R1 | (-0.962, -0.242, +0.124) |
| Norm of R1 before normalization | 1.014 (limit 0.9 to 1.1: pass) |
| Angle of R1 to (-1, 0, 0) | 15.8 deg on the Poincare sphere |
| Pair imbalance | 0.023, -0.190, 0.139 (limit 0.15: the +45'/-45' pair fails) |
| Agreement of the d values that constrain d | 0.088 (limit 0.15: pass) |

The x'/y' pair is almost pure lab y / lab x, so it carries no information on d; its d value is not meaningful.

### 3e. Two foci at stage 0 and 90 deg (one exposure of 24, no saturation)

| | (a) weighted TM, x'/y', no polarizer information | (b) weighted TM, U from 3d | (c) blind TM, x'/y' |
|---|---|---|---|
| Where the foci appear | A at stage 90, B at stage 0 | A at stage 0, B at stage 90 | A at stage 90, B at stage 0 |
| Pass fraction at stage 0, raw (A, B) | 0.003, 0.965 | 0.963, 0.002 | 0.013, 0.714 |
| Separation, raw | -0.962 | +0.962 | -0.701 |
| Pass fraction before the arm (A, B) | 0.018, 0.993 | 0.993, 0.009 | 0.067, 0.932 |
| Separation before the arm | -0.976 | +0.984 | -0.866 |
| Leak of the blocked focus at stage 0 | 0.009 | 0.005 | 0.032 |
| Leak of the blocked focus at stage 90 | 0.014 | 0.013 | 0.164 |
| PBR of focus A at stage 0 / 90 | 11 / 358 | 653 / 8 | 21 / 324 |
| PBR of focus B at stage 0 / 90 | 489 / 8 | 10 / 397 | 499 / 42 |
| Scan: azimuth, degree of linear polarization, focus A | 90.9 deg, 1.00 | 172.8 deg, 1.01 | 78.8 deg, 1.04 |
| Scan: azimuth, degree of linear polarization, focus B | 11.1 deg, 1.00 | 90.5 deg, 1.07 | 150.4 deg, 0.72 |

- Leak = power of the blocked focus / power of the passed focus at that stage angle.
- Using d = -0.701 (minus the learned b) instead of -0.693 changes the separations before the arm by 0.001.
- The scans of (a) and (c) were recorded 6 minutes later, during 3f. The Malus fits have residuals of 6 to 10 %,
  so degrees of linear polarization slightly above 1 are fit scatter.
- From the scan of (a): with the polarizer axes at stage 6 and 96 deg the raw separation would be -0.986; this is the
  largest value, and the offset equals theta0 of 3c.

### 3f. Foci at stage 45 and 135 deg with the phase of 3c (optional; the phase was stable)

Designed with the speckle frame only (U with its phase, d = -0.706, theta0 = 6.0 deg), no foci calibration.

| Variant | Pass fraction at stage 45 (A, B) | Separation 45 / 135 | Predicted | Azimuth A, B | Predicted |
|---|---|---|---|---|---|
| Linear at the polarizer (behind the arm) | 0.999, 0.004 | +0.995 | +1.000 | 46.9, 133.3 deg | 45.1, 134.8 deg |
| Linear before the arm | 0.849, 0.147 | +0.702 | +0.712 | 68.2, 110.1 deg | 68.8, 114.3 deg |

## The two tests of the hypothesis

1. **U is a swap of the two labels.** From speckle: cross-term power 0.01 %, x' linear (ellipticity 0.0 deg), in axes
   turned by 6.0 deg from the stage zero. From foci: R1 lies 15.8 deg from (-1, 0, 0) on the Poincare sphere, which is
   a turn of the axes by 7 deg in real space and an ellipticity of 3.6 deg; in the stage axes the cross-term power
   is 1.9 %. The scan of (a) puts the best polarizer axes at 6.0 deg. So the axes of the TM are the stage axes up to a
   swap and an offset of 6 to 7 deg of the stage zero. The data of 12 Nov showed the same offset (favoured axis at 94 to 95 deg).
2. **d equals minus b.** d from the foci is -0.693 (-0.697 from the two pairs that constrain it); from speckle
   -0.706, 0.699 and -0.707; the learned b is +0.701. They agree within 0.01.

## Times

| Event | Time | After the dataset |
|---|---|---|
| Dataset | 10:43 | |
| 3a to 3c (speckle frames) | 12:02 to 12:04 | 79 to 81 min |
| 3d (foci calibration, tests (a) and (c)) | 12:05 | 82 min |
| 3e (test (b) and its scan) | 12:06 | 83 min |
| 3f (foci at 45 / 135 deg, scans of (a) and (c)) | 12:11 to 12:12 | 89 min |

## Code

No protected file was changed (`run_tm_focus_optimization.py`, `create_refs.py`, `tm_system_checkpoint.pt`,
`measurement_focus.h5`: sizes and times checked before and after). `TM_retrieve.py`, `run_polarization_axes.py` and
`polarizer_arm_estimator.py` are unchanged too.

- `TM_retrieve_camweights.py` (my copy of `TM_retrieve.py`): `enable_learnable_fourier_weights(b_init)` adds the
  parameter beta, and the Fourier weights become (1 + b, 1 - b) with b = tanh(beta). The image camera stays at (1, 1).
  The Stage A version is saved as `tm_focus_outputs/stageA_offline/TM_retrieve_camweights_as_used_stageA.py`.
- New: `stageB_step1_check.py` (step 1), `retrieve_tm_learn_b.py` (step 2), `stageB_axes_agreement.py`
  (axes of independent retrievals), `run_polarizer_test.py` (step 3), `stageB_collect_report_numbers.py`.

## Output files

- `tm_focus_outputs/tm_system_checkpoint_camweights_learned_b_20261115.pt`
- `tm_focus_outputs/stageB_offline/`: `step1_check.json`, `learn_b_*.json` (four runs, with b per epoch),
  `axes_agreement.json`, logs.
- `tm_focus_outputs/polarizer_test_20261115_1201/`:
  - `stageB_report.md` (this file), `polarizer_test_summary.json`, `report_numbers.json`, logs,
    `run_polarizer_test_as_run_abcde.py`
  - `speckle_frames_main_angles.npz` (104 masks at 0/45/90/135 deg), `state_step_a.npz` (crop offsets)
  - `jones_matrix_U_from_foci.npy`
  - per pattern, `two_foci_<tag>_phase_rad.npy`, `_phase_uint8.npy`, `_full_slm_mask_uint8.npy`, `_phase_hsv.png`,
    `_frames.npz`, `_overview.png`, and `im_two_foci_<tag>_pol_0deg.png`, `_pol_90deg.png`. Tags:
    `probe_xy_weighted_tm` (test (a)), `probe_pm45_weighted_tm`, `probe_RL_weighted_tm`, `xy_blind_tm` (test (c)),
    `lab_axes_U_from_foci` (test (b), with the scan), `pm45_at_polarizer`, `pm45_before_arm`,
    and the scans `scan_probe_xy_weighted_tm`, `scan_xy_blind_tm`.

## Addendum, 15 Nov 13:05: follow-up stopped, and a provisional estimate of the extinction angles

**Follow-up (lab x at stage 6 deg, fine scan of pattern (a)) was not recorded.** The freshness check at 13:03
(140 min after the dataset) gave a mean correlation of 0.013 for the 8 masks (-0.04 to 0.06); it was 0.945 at 12:12.
The run stopped there, as instructed. The drop is abrupt, not gradual ageing; its cause is not known. The run saved
only the correlation values at that step, so it cannot tell missing light from a changed speckle pattern
(`run_polarizer_followup.py` now saves the frames and their mean level at the freshness step).
Folder: `tm_focus_outputs/polarizer_followup_20261115_1303/`.

**Provisional extinction angles of pattern (a), from the coarse scan of 12:12** (15 deg steps, exposure 24;
not the requested fine scan):

| Fit | Focus A | Focus B |
|---|---|---|
| Points within 38 deg of the minimum, 3x3 window | +3.7 deg | 99.6 deg |
| All 12 angles, 3x3 window | +0.9 deg | 101.1 deg |
| Points within 38 deg of the minimum, 5x5 window | +3.4 deg | 98.5 deg |
| All 12 angles, 5x5 window | +0.5 deg | 100.0 deg |

The statistical errors of the fits are 0.3 to 1.4 deg, but the fits differ by up to 2.8 deg, so the uncertainty is
about 2 deg per focus.

| Quantity | Local fits | All angles |
|---|---|---|
| Angle between the two polarization directions behind the arm | 95.9 deg | 100.2 deg |
| Before the arm, d = -0.70, arm axes at stage 0 | 85.4 deg | 92.7 deg |
| Before the arm, d = -0.70, arm axes at stage 6 deg | 97.1 deg | 104.2 deg |

Dividing out the arm multiplies an error in the direction of focus A by 2.3, so these angles are uncertain by about 5 deg.
With arm axes at stage 0 the two directions are compatible with 90 deg; with arm axes at stage 6 deg they are 7 to 14 deg
from 90 deg. They would be exactly 90 deg apart for arm axes at about stage -1 to +2 deg. This is in tension with
theta0 = +6.0 deg of the speckle estimate (step 3c) and is not settled by the coarse scan.
Files: `coarse_scan_extinction_estimate.json`, `log_coarse_scan_extinction_estimate.txt`.
