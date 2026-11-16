# Polarization-frame tests and PBR background check

Scripts written on the lab PC in November 2026. They produced the results of Supplementary Section S7 (camera weight of the beam splitter, polarization frame of the retrieved TM) and the check of the background level of the PBR.

Like the other files in `lab/`, they cannot run for a reader. They need the bench (SLM, two Basler cameras, a linear polarizer on a Thorlabs KDC101 rotation mount in front of the Fourier-plane camera), `pypylon`, `thorlabs_apt`, and the private lab modules `create_refs` and `run_tm_focus_optimization`.

The scripts import `TM_retrieve_camweights`, the lab copy of the forward model with the camera weight. In this repository the same change is part of `TM_retrieve.py` (`fourier_pol_weights`, `enable_learnable_fourier_weights`).

Results are in `results/polarization/` and `results/pbr_bias/`. Angles in the scripts and reports are stage angles. The reports call stage 0 or 6 degrees "lab x"; with the Fourier-plane camera on the transmitted port of the beam splitter, the in-plane (p) axis is at stage 94.9 degrees.

| Script | Role |
|---|---|
| `retrieve_tm_learn_b.py` | Joint fit of the TM with the camera weight b = tanh(beta) learned, or held fixed. |
| `retrieve_tm_camera_weights.py` | Joint fit with a fixed b; shared offline helpers (`load_data`, `build_system`, `held_out`). |
| `stageB_step1_check.py` | Check of a dataset (speckle contrast, levels) and of the standard TM (held-out correlation). |
| `stageB_axes_agreement.py` | Compares the polarization axes of several retrieved TMs. |
| `run_polarization_axes.py` | Library of the tests: stage class, image registration, frame estimate from polarizer speckle frames, two-foci optimizer, focus metrics. |
| `polarizer_arm_estimator.py` | Estimators for the field at the polarizer with unequal weights (frame from speckle frames, frame from pairs of foci). |
| `run_polarizer_test.py` | Hardware: frame from speckle frames at four polarizer angles, calibration with foci, two-foci tests, foci at 45 and 135 degrees. |
| `run_polarizer_followup.py` | Hardware: 2 degree scans at three exposures, full scan, pairs of orthogonal linear foci. |
| `run_polarizer_followup_b.py` | Hardware: repeat of one two-foci test after the foci calibration. |
| `stageB_collect_report_numbers.py`, `stageB_followup_analysis.py` | Offline numbers of the two reports (extinction angles, axes of the weights, low-signal response). |
| `find_static_polarizer_angle.py` | Hardware: stage angle at which the rotating polarizer is parallel to a fixed polarizer. |
| `collect_best_two_foci_images.py` | Hardware with `--capture`: polarizer images of a two-foci pattern and an image without polarizer. |
| `move_by.py` | Rotates the mount by a relative angle. |
| `stageA_offline_analysis.py` | Analysis of the first dataset of 12 Nov. That dataset was recorded with a polarizer left in front of the image-plane camera; its numbers are superseded. |
| `run_pbr_bias_check.py` | Hardware: single foci recorded at four exposures to measure the bias of a single-frame PBR. |
| `pbr_bias_hdr_analysis.py`, `dark_level_analysis.py` | Offline analysis of the dark level of the cameras and of its effect on the PBR. |
