# Polarization-frame tests (12 to 15 Nov 2026)

Reports, summary files and camera frames of the rotating-polarizer tests behind Supplementary Section S7. The scripts are in `lab/polarization/`. Absolute paths of the lab PC were replaced by `<lab>`.

| Folder | Content |
|---|---|
| `stageB_offline/` | Dataset check and the joint fits with the camera weight b (`learn_b_*.json`: b, training loss and held-out loss per epoch). |
| `polarizer_test_20261115_1201/` | Polarizer test of the TM with the fitted weight: report `stageB_report.md` and summaries. |
| `polarizer_followup_20261115_1342/` | Follow-up: 2 degree scans, pairs of orthogonal foci, axes of the weights: `followup_report.md`. |
| `static_polarizer_20261115_1450/`, `static_polarizer_result.md` | Stage angle of a fixed polarizer aligned with a laboratory axis. |
| `figure_frames/` | Camera frames in counts (16-bit PNG) used by `paper/make_polarization_figure.py`; see its `README.md`. |
| `best_two_foci_three_images/` | Information on three images of a two-foci pattern with and without polarizer. |
| `standard_pipeline_run_20261115/` | Summary of the standard run that recorded the dataset of 15 Nov. |
| `polarization_axes_*_run1/`, `stageA_offline/`, `polarizer_followup_20261115_1303/`, `_1312/` | Earlier runs, kept as a record. The runs of 12 Nov used a dataset recorded with a polarizer left in front of the image-plane camera and are superseded. |

Angles are stage angles of the rotating polarizer. The reports call stage 0 or 6 degrees "lab x". With the Fourier-plane camera on the transmitted port of the beam splitter, the in-plane (p) axis is at stage 94.9 degrees.

Frames recorded at the shortest exposure (24 microseconds) read low signals about one count too low, because the dark level of the camera is clipped at zero (`results/pbr_bias/`).
