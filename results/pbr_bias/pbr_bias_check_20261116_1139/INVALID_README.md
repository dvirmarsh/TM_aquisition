# This run is INVALID as a PBR measurement: the SLM did not modulate the light

Run of 16 Nov 2026, 11:38 to 12:06, `run_pbr_bias_check.py --allow-new-dataset`.

What the frames show:

- Every camera frame is the same speckle, whatever mask was sent to the SLM.
  Correlation between focus frames of different target patterns: 0.993; between repeated frames of one pattern: 0.994.
  New dataset: frame 0 against frames 1 to 6031: 0.975 to 0.999 (image plane), 0.992 to 1.000 (Fourier plane).
  In the dataset of 15 Nov the same correlations are about 0 (image plane) and 0.1 to 0.16 (Fourier plane).
- The light level at the image camera is 2.6 to 2.8 times the level of 15 Nov at the same exposure, and nothing is seen
  outside the fiber core, so it is light through the fiber, not stray light.
- No focus was formed: peak 139 to 229 counts on a background of 29 counts at exposure 24 (the TM predicted PBR 500 to 1000).

Consequences:

- The PBR numbers and ratios in `pbr_bias_check_summary.json` and in `log.txt` are NOT a measurement of the PBR bias.
- The dataset `TM_lib/measurement_focus_pbr_check_20261116_1139_INVALID_slm_not_modulating.h5` and the TM
  `tm_system_checkpoint_pbr_check_INVALID_slm_not_modulating.pt` recorded in this run are useless (6032 times the same frame).
  They were renamed with the suffix INVALID after the run; the summary JSON still holds the names without the suffix.
  On 16 Nov, at Dvir's request, both files were moved to the Windows Recycle Bin.
- Still valid: the camera settings read from the cameras (`camera_state_before`, `camera_state_after`) and the frames as a
  record of how the image camera responds to the same static speckle at exposures 24, 96, 384 and 1536.

Software side at the time (read-only checks at 12:20): Windows not locked, two displays active (1920 x 1080 primary and
1920 x 1200, the SLM), Blink SDK constructed without error. The cause was not found from the PC.
