# Stage A report (offline, no hardware), 12 Nov 2026

Data: the 12:34 dataset (`measurement_focus.h5`), the current TM (`tm_system_checkpoint.pt`), and the saved runs
`polarization_axes` (image plane, 16:01) and `polarization_axes_fourier` (Fourier plane, 16:22).
`polarizer_arm_estimator.py` prints "self-test passed".

Conventions: correlation = Pearson correlation of intensities. tau = P_0 / (P_0 + P_90) is the fraction of a focus
that passes the polarizer at stage 0. Separation = tau_A - tau_B. "Before the arm" = powers divided by (1 + d, 1 - d).

## Main results

1. The polarizer does not degrade the image: I_0 + I_90 matches the 12:34 Fourier frames better (0.90) than the
   image plane matches itself over the same time (0.85).
2. d from the saved foci is reproduced (-0.718 and -0.627, mean -0.673). The mean light gives -0.695 independently.
3. With the current TM, the speckle estimate with diattenuation finds the frame but not d. Its three d values
   disagree (-0.98, 0.88, -0.39). With its frame and d = -0.673 from the foci, the four saved foci are predicted
   to within 0.013 on average.
4. Camera weights in the retrieval help. The held-out correlation rises from 0.79 / 0.83 (image / Fourier) at b = 0
   to 0.88 / 0.94 at b = 0.7 and falls again at 0.8 and 0.9. With the b = 0.7 TM the frame is aligned with the stage
   axes (x' at 93.7 deg, ellipticity 0.0 deg) and the speckle estimate alone predicts the saved foci to within 0.04.
5. Open point: relative to the b = 0.7 TM the arm still shows d between -0.83 and -0.96, more than b = 0.7.

## A1. Effect of the polarizer on the image

| Quantity | Value |
|---|---|
| Fourier plane, I_0 + I_90 (registered) against the 12:34 frame, four masks, r <= 50 px | 0.881, 0.916, 0.923, 0.892 (mean 0.903) |
| Fourier plane, I_45 + I_135 against the 12:34 frame | 0.90 to 0.92 (mean 0.910) |
| Image plane, 16:01 against 12:34, same masks, core | 0.82 to 0.87 (mean 0.849) |
| Level of I_0 + I_90 relative to the 12:34 frame, same exposure (675) | 0.78 (0.76 to 0.81) |
| Displacement from the 12:34 frames (no model) against the run's values, 0/45/90/135 deg | differences 0.16, 0.05, 0.03, 0.01 px |
| Speckle contrast of the 12:34 Fourier frames (6032 masks, NA disk, mean envelope divided out) | 0.82 |
| Speckle contrast of the 12:34 image frames (6032 masks, core, mean envelope divided out) | 1.03 |
| Reference: Fourier camera behind the polarizer, one angle (four masks) | 1.04, 0.96, 0.96, 0.95 at 0/45/90/135 deg |
| Reference: Fourier camera without polarizer, the same four masks | 0.81 |
| Mean light behind the polarizer at 0/45/90/135 deg | 71, 155, 243, 155 counts: 77.4 % along stage 90, degree of linear polarization 0.55, azimuth 90.0 deg |

The Fourier contrast of 0.82 is what a 77 : 23 split predicts (0.81). The image-plane contrast equals that of a
single polarization on the Fourier camera. The probe masks are not random patterns, so this is a hint, not a proof,
that the image camera records mostly one polarization.

## A2. Saved foci

| Quantity | Uncorrected pattern | Corrected pattern |
|---|---|---|
| d from diattenuation_from_orthogonal_pair | -0.718 | -0.627 |
| Pass fraction at stage 0, raw (A, B) | 0.160, 0.123 | 0.668, 0.026 |
| Separation, raw | +0.037 | +0.643 |
| Pass fraction before the arm (A, B), d = -0.673 | 0.493, 0.418 | 0.911, 0.118 |
| Separation before the arm | +0.075 | +0.793 |

Using each pattern's own d changes the separations before the arm by 0.002.
Independent check: the current TM needs d = -0.695 to reproduce the 77 : 23 split of the mean light.

## A3. Speckle estimate with diattenuation, current TM (16 saved frames, r <= 50 px)

| Quantity | Value |
|---|---|
| d from the S0 term of I_0 - I_90 | -0.98 (the estimator's limit) |
| abs(d) from the amplitude of I_45 - I_135 | 0.878 |
| d from the polarization dependence of I_0 + I_90 | -0.388 |
| theta0 | -2.9 deg |
| Explained fraction | 0.49 (was 0.04) |
| Correlation per angle, 0/45/90/135 deg | 0.559, 0.576, 0.713, 0.599 (was 0.494, 0.537, 0.641, 0.625) |
| Angle between the first row of this R and that of the run's R | 33 deg on the Poincare sphere |

Pass fractions of the four saved foci at stage 0 (uncorrected A, uncorrected B, corrected A, corrected B):

| Source | Values | Mean error |
|---|---|---|
| Measured | 0.160, 0.123, 0.668, 0.026 | |
| Estimator with its own d (-0.98) | 0.020, 0.004, 0.090, 0.005 | 0.214 |
| Estimator's frame with d = -0.673 from the foci | 0.186, 0.148, 0.666, 0.025 | 0.013 |
| Unitary U of the 16:22 run | 0.441, 0.564, 1.000, 0.000 | 0.270 |

The 33 deg error of the run's frame predicts a separation before the arm of cos 33 deg = 0.84 for the corrected
pattern; the measured value is 0.79.

## A4. Camera weights in the forward model

Check of b = 0: the copied code reproduces the forward model of the current checkpoint exactly (largest difference 0).
Held-out correlation: mean over the 302 held-out masks, full 128 x 128 frame. Settings of the original run
(100 epochs, learning rate 0.01, batch 512, both planes, same scale between the planes), 11.5 minutes per retrieval.

| b | Image plane | Fourier plane |
|---|---|---|
| Current checkpoint | 0.792 | 0.829 |
| 0 | 0.792 | 0.832 |
| 0.2 | 0.822 | 0.865 |
| 0.4 | 0.850 | 0.894 |
| 0.6 | 0.869 | 0.922 |
| 0.7 | 0.881 | 0.937 |
| 0.8 | 0.855 | 0.919 |
| 0.8, second seed | 0.861 | 0.910 |
| 0.9 | 0.852 | 0.924 |
| -0.4 | 0.851 | 0.895 |

b = 0.7 and 0.9 and the second seed of 0.8 were added to locate the maximum. b = -0.4 equals b = +0.4, so the
retrieval finds the sign by itself. Caveat: the runs with b >= 0.6 were still improving at epoch 100, and the
training at b = 0.9 became unstable, so the position of the maximum is known to about 0.1.

Kept: `tm_focus_outputs/tm_system_checkpoint_camweights_b0.70.pt` (loading it needs `TM_retrieve_camweights.py` in TM_lib).

### A3 repeated on the b = 0.7 TM

| Quantity | Value |
|---|---|
| d (three estimates) | -0.902, 0.958, -0.828 |
| d needed to reproduce the mean light | -0.84 |
| theta0 | +0.8 deg |
| Explained fraction | 0.66 |
| Correlation per angle, 0/45/90/135 deg | 0.655, 0.753, 0.882, 0.752 |
| abs(U) | [[0.06, 1.00], [1.00, 0.06]] |
| x' of the TM at the polarizer | azimuth 93.7 deg, ellipticity 0.0 deg |
| Saved foci predicted (same order as above) | 0.175, 0.051, 0.646, 0.069 (mean error 0.038) |

U is diagonal up to a swap of the axis labels: x' (the favoured component, weight 1.7) arrives along stage 90 and
y' along stage 0; the cross terms carry 0.4 % of the power. The b = -0.4 TM gives the unswapped form
(abs(U) = [[1.00, 0.02], [0.02, 1.00]]).

## Code and files

No existing file was changed. `run_tm_focus_optimization.py`, `create_refs.py`, `TM_retrieve.py`,
`run_polarization_axes.py`, `tm_system_checkpoint.pt` and the saved run folders are untouched.

New files in TM_lib:
- `TM_retrieve_camweights.py`: copy of `TM_retrieve.py`; the Fourier camera records w_x |E_x|^2 + w_y |E_y|^2 with
  `fourier_pol_weights` (default (1, 1)). Five places differ from the original.
- `retrieve_tm_camera_weights.py`: step A4 (check of b = 0, one retrieval per call).
- `stageA_offline_analysis.py`: steps A1 to A3.

Outputs in `tm_focus_outputs/stageA_offline/`:
- `stageA_report.md` (this file)
- `stageA_current_tm.json`, `log_A1_A2_A3_current_tm.txt`
- `camweights_check_b0.json`, `camweights_b*.json`, `log_camweights_gpu0.txt`, `log_camweights_gpu1.txt`
- `stageA_camweights_b*.json`, `log_A3_camweights_b*.txt`
- `log_mean_light_check.txt`
