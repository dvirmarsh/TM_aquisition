# Static polarizer: angle of the rotating polarizer that is parallel to it

Measured 15 Nov 2026, 14:50:25 to 14:53:37, with `code/find_static_polarizer_angle.py`.
No report was written at the time; this page is put together from
`static_polarizer_20261115_1450/static_polarizer_summary.json` (keys in brackets) and `log.txt`.

Setup: a static polarizer, placed by Dvir along one lab axis (x or y, within a few degrees), in front of the rotating
polarizer, which sits in front of the transmission Fourier-plane camera (40238131). One fixed probe mask on the SLM.

## Result

| Quantity | Value | Source |
|---|---|---|
| Rotating polarizer parallel to the static one | stage **95.13 deg** (and 275.13 deg) | `fine.aligned_angle_deg` = 95.1294 |
| Crossed (extinction) | stage 5.13 deg (measured at 185.13 deg) | `fine.extinction_angle_mod_180_deg` = 5.1294 |
| Uncertainty | 0.005 deg (statistical 0.004 deg; spread over the exposures 0.006 deg) | `fine.sigma_deg`, `fine.sigma_statistical_deg`, `fine.spread_over_exposures_deg` |
| Downward scan minus upward scan | +0.001 deg | `fine.down_minus_up_deg` |
| Power at the maximum / lowest power of the fine scan | 10754 | `log.txt`, section "Check of the maximum" |
| Stage left at | 95.13 deg | `stage_left_at_deg` |

The uncertainty is that of the fit only. It does not include how well the static polarizer itself is aligned to the
lab axis.

## Steps

| Step | Angles | Exposure | Result | Source |
|---|---|---|---|---|
| Coarse scan | 0 to 170 deg, steps of 10 deg | 675 | Malus fit: maximum at 95.06 deg, fitted degree of linear polarization 1.014, measured minimum / maximum 0.0043 | `coarse.malus_fit`, `coarse.minimum_over_maximum_measured` |
| Fine scan, upward and downward | 173 to 197 deg, steps of 1 deg | 675 | extinction 185.140 deg (up), 185.130 deg (down) | `fine.fits.675` |
| | | 2700 | 185.131 deg (up), 185.132 deg (down) | `fine.fits.2700` |
| | | 10800 (used) | 185.129 deg (up), 185.130 deg (down) | `fine.fits.10800` |
| | | 43200 | saturated in the range, not used | `log.txt` |
| Check of the maximum | 75 to 115 deg, steps of 5 deg | 675 | parabola maximum at 94.99 deg, 0.14 deg from extinction + 90 deg | `maximum_check` |

- The fine scan was taken 180 deg from the coarse extinction angle because the range would otherwise cross stage 0.
- Power = sum of the camera counts over a disk around the image, minus the dark level (median far from the image).
- One warm-up frame was discarded after every change of stage angle, SLM pattern or exposure (228 in total).
- Pixel format Mono12. Frames were not saved; the JSON holds the power at every angle (`coarse.power`,
  `fine.fits.<exposure>.power_up`, `power_down`, `maximum_check.power`).
- For comparison: the strong axis of the arm found with the pairs of foci is at stage 94.9 deg
  (`polarizer_followup_20261115_1342/followup_report.md`), 0.2 deg from this angle.
