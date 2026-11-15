# Measurement data

The repository ships pointers only. The two published H5 files (~1.4 GB together) are on Zenodo: https://doi.org/10.5281/zenodo.24243174. `data/download_data.py` downloads them and verifies their sha256.

| File | Size | sha256 |
|---|---|---|
| `measurement_motor_pcc.h5` | 995 080 280 B | `570712f3a5a030466faa553a4c1a76e9b3daed5589bb5c358e7fa5024a0fe814` |
| `measurement_raf.h5` | 404 550 496 B | `0ab85ff053df6139c524cb12308fd32d372f8d55fa4bbbc38aa797f636c48e4b` |

```bash
python data/download_data.py          # download and verify
python data/download_data.py --force  # redownload even if present
```

`mode_fields/` is git-ignored derived data. Mode fields are regenerated on demand and must never be uploaded.

## What is in each file

- `measurement_motor_pcc.h5` — modal pipeline. Groups `initial_calibration` and `post_motor`. Each group has `PhaseMask (6032, 128, 128) uint8`, `transmission_image (6032, 128, 128) uint16`, `transmission_fourier (6032, 128, 128) uint16`. The two groups are the same fiber before and after a motorized displacement: aberration calibration is fit on `initial_calibration`, then the TM alone is retrieved on `post_motor`.
- `measurement_raf.h5` — pixel-basis baselines. One group `experiment_0`, with `PhaseMask (6032, 32, 32) uint8` (coarser SLM macro-pixels) and the same two 128×128 uint16 camera stacks. Consumed by `baselines/run_raf_offline.py` and `baselines/run_admm_offline.py`, which use the image plane only.

Group names in use: `initial_calibration` (`motor_pos_utils.CALIBRATION_H5_GROUP`), `post_motor`, `experiment_0`.

## Conventions a reader cannot guess

- `PhaseMask` / `slm_phase_mask` is `uint8`, mapped as `phase = value * 2π / 256`.
- SLM patterns and image-plane intensities are flipped on **both** spatial axes (`np.flip(..., axis=(-2, -1))`) to undo the 4F system and beamsplitter. The Fourier plane is **not** flipped.
- The image plane is rescaled by `im_factor` so the two planes share one intensity scale:

  `im_factor` = magnification² · (bs_split / (1 − bs_split)) · (t_image / t_fourier)

  In `run_modal_tm_retrieval.py`, magnification = 50 and `bs_split = 0.47736`; the exposure times come from the runner config (`transmission_image_et`, `transmission_fourier_et`). See `tran_im_factor` in `run_offline`.

## Loaders

- `utils.load_real_mmf_data` — train/test split, fuzzy key matching. Used by the Adam calibration stage.
- `run_modal_tm_retrieval.load_all_mmf_data` — whole group, no split. The subset benchmarks use this and do their own deterministic index sampling via `motor_pos_utils.sample_random_subset_indices(n, size, trial, seed)`.
- `utils.list_h5_experiments(path)` — prints the group names in a downloaded H5 file.

## Fiber and grid that key the mode-field cache

These are the numbers used in the published runners (do not invent others):

- core radius `a = 25e-6` m
- core index `n1 = 1.457`
- numerical aperture `NA = 0.22` (cladding `n0 = sqrt(n1² − NA²)`)
- wavelength `λ = 632.8e-9` m
- grid 128×128
- guided-mode count `M = 1508`

Caches are written as `mode_fields_<md5[:8]>.pt`, keyed by a hash of fiber + grid parameters (`utils.get_mode_fields_filename`). Changing core radius, index, wavelength, or grid silently produces a new multi-hundred-MB cache file. Paper scripts write the cache in the current working directory; `demo.py` writes under `data/mode_fields/`.
