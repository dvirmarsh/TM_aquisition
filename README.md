# Reference-less 3D vectorial transmission-matrix retrieval in multimode fibers

**Breaking the Phase-Ambiguity Limit: Reference-less 3D Vectorial Transmission-Matrix Retrieval in Multimode Fibers**

Dvir Marsh, Lior Fridman, Shie Mannor, Guy Bartal  
Andrew and Erna Viterbi Department of Electrical & Computer Engineering, Technion, Haifa 3200003, Israel

Reference-less retrieval of a multimode fiber's transmission matrix (TM) in the guided-mode basis, from intensity-only camera measurements (no interferometer). One retrieved TM controls two output planes (image and Fourier) and both polarization components.

Manuscript: `paper/main.tex`. Data DOI: [10.5281/zenodo.24243174](https://doi.org/10.5281/zenodo.24243174). Citation: see `CITATION.cff`.

```
Marsh, D., Fridman, L., Mannor, S. & Bartal, G.
Breaking the Phase-Ambiguity Limit: Reference-less 3D Vectorial
Transmission-Matrix Retrieval in Multimode Fibers.
```

## What is in this repository

```
mode_solver.py, utils.py, TM_retrieve.py, TM_retrieval_algos.py,
metrics_losses.py, motor_pos_utils.py
    Core library, flat at the repo root (the lab parent imports this
    checkout as TM_lib/).

run_modal_tm_retrieval.py
    Flagship offline runner: Adam calibration, then ADMM retrieval.

demo.py
    CPU synthetic demo; no data download.

data/                  Published H5 pointers + download script. Mode-field
                       caches in data/mode_fields/ are git-ignored.
baselines/             RAF2-1 and scalar ADMM pixel-basis baselines.
lab/                   Hardware acquire / wavefront-control / replay. Cannot
                       run without the private parent repo and the bench.
paper/                 Manuscript, figures, and figure scripts.
results/               Small CSVs/JSON/MD consumed by the figure scripts.
tests/                 Import smoke + a small synthetic ADMM test.
LICENSE, CITATION.cff, requirements.txt, .gitignore
```

`TMMeasurementSystem` exposes the forward model only. Exploratory plotting, evaluation helpers, and the unused symmetric-TM path were removed for this release.

Target tree (directories marked git-ignored exist on disk but do not enter git):

```
README.md  LICENSE  CITATION.cff  requirements.txt  .gitignore
mode_solver.py  utils.py  TM_retrieve.py  TM_retrieval_algos.py
metrics_losses.py  motor_pos_utils.py
run_modal_tm_retrieval.py  demo.py
data/{README.md, manifest.json, download_data.py, measurement_*.h5, mode_fields/}
baselines/{README.md, raf21_tm.py, admm_tm.py, run_raf_offline.py, run_admm_offline.py}
lab/{README.md, acquire_*.py, run_wavefront_control.py, replay_*.py}
paper/{main.tex, references.bib, figures/, image_results/, make_*.py}
results/{README.md, offline/{modal,raf21,admm}/, lab_replay/}
tests/test_smoke.py
```

## Three-phase workflow

```
lab/ acquire  -->  data/*.h5  -->  run_modal_tm_retrieval.py / baselines/
                         |                    |
                         |                    v
                         |            SLM focus masks (.h5)
                         |                    |
                         v                    v
                   lab/ replay  <--  project masks on the real SLM
                         |
                         v
              results/lab_replay/*.csv  -->  paper/make_focus_vs_time_figure.py
```

1. **Acquire** (lab, hardware). Project SLM patterns, record image- and Fourier-plane frames. Needs `pypylon` and `create_refs` from a private parent repository. A reader cannot run this; see `lab/README.md`.
2. **Retrieve** (offline). Read the H5 file, calibrate the forward model, retrieve the TM, write focus-maximizing SLM masks. This is what a reader with the published data can rerun.
3. **Validate** (lab, hardware). Project those masks, measure the focus, write CSVs. Those CSVs are Fig. `focus_time`.

## Installation

Python 3.12. The checked-in `.venv` is Python 3.12.3 with `torch==2.7.1+cu126`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Paper scripts default to `--device cuda`. On CPU they run but are much slower. Realistically CPU-runnable: `demo.py`, `tests/test_smoke.py`, and the two `paper/make_*.py` figure scripts. The offline retrieval runners (`run_modal_tm_retrieval.py`, `baselines/run_*_offline.py`) are written for a GPU; paper-scale runtimes on this machine were **not measured**.

## Getting the data

Two files (~1.3 GB). sha256 values are in `data/manifest.json`.

```bash
python data/download_data.py
```

This downloads `measurement_motor_pcc.h5` and `measurement_raf.h5` from Zenodo ([10.5281/zenodo.24243174](https://doi.org/10.5281/zenodo.24243174)) into `data/` and verifies their sha256.

| File | Role |
|---|---|
| `data/measurement_motor_pcc.h5` | Modal pipeline. Groups `initial_calibration` and `post_motor`, each with 6032 probes at 128×128. |
| `data/measurement_raf.h5` | Pixel-basis baselines. Group `experiment_0`, 32×32 SLM macros, 128×128 cameras. |

`PhaseMask` is `uint8` mapped as `phase = value * 2π/256`. SLM patterns and image-plane intensities are flipped on both spatial axes; the Fourier plane is not. The image plane is rescaled by `im_factor` = mag² · (bs_split / (1 − bs_split)) · exposure ratio so both planes share one intensity scale (`bs_split = 0.47736`, mag = 50 in the modal runner). Full conventions: `data/README.md`. Inspect a file with `utils.list_h5_experiments("data/measurement_motor_pcc.h5")`.

## Quickstart

```bash
python demo.py
python demo.py --quick          # smallest settings that still fit the synthetic data
```

`demo.py` builds a ~60-mode fiber on a 32×32 grid, synthesizes dual-plane intensities from a known TM (`sample_measurements_slm`), runs Algorithm 1 (ADMM, `init_method='default_beta_prior'`), and reports the Pearson correlation between predicted and measured intensities in both planes. It then refits on the image plane only and scores the unseen Fourier plane — the cross-plane transfer of the manuscript, on synthetic data. No H5, no GPU. `--quick` finished in a few seconds on CPU in this checkout. Output figure: `results/demo/demo_summary.png`.

The demo deliberately does not report a distance between the retrieved and ground-truth matrices. The SLM modulates one polarization on a coarse grid, so the probes span only part of the mode space (rank 30 of 60 modes here) and the unexcited part of the TM is not determined by any measurement; a whole-matrix error is therefore not a meaningful score. What the retrieval must reproduce is the field, which is what the reported correlations measure.

### Reproduce the paper's TM retrieval

Data must be in `data/`. Commands below were **not** timed here (GPU paper-scale runs).

**Step 1 — Adam joint calibration** on `initial_calibration` (Algorithm 2). Fits the TM and the aberration/geometry nuisance parameters. Writes `results/offline/modal/tm_system_checkpoint.pt`. There is no published checkpoint: `--use-tm-checkpoint` defaults to False, so step 1 regenerates it. Runtime: **unmeasured**.

```bash
python run_modal_tm_retrieval.py --h5-path data/measurement_motor_pcc.h5 --method admm
```

**Step 2 — `calibrate()` + `validate_aberrated_mode_basis()`.** The runner calls these after loading or fitting the checkpoint. `calibrate()` pre-applies the aberration chain to each basis mode and **resets the TM** via `reinitialize_optimized_tm()`. Failed validation means the cached aberrated mode basis and the live parameters disagree.

**Step 3 — ADMM subset benchmark** on `post_motor`. After every epoch the runner optimizes a focus-maximizing SLM mask into `results/offline/modal/slm_focus_masks.h5`. Subset sizes are 1.2M, 1.5M, 2M, 3M with M = 1508. Runtime: **unmeasured**.

Baselines (image plane only, `data/measurement_raf.h5`):

```bash
python baselines/run_raf_offline.py
python baselines/run_admm_offline.py
```

They write `results/offline/raf21/` and `results/offline/admm/` (`admm_benchmark_*` tables). Runtimes: **unmeasured**.

Useful flags on the modal runner (verified via `--help`): `--method admm` (the only published choice), `--use-tm-checkpoint` (default False), `--step3-mode pcc-focus`, `--device cuda|cpu`, `--h5-path`, `--output-dir` (default `results/offline/modal`).

The ADMM call the runner uses is:

```python
history = optimize_tm_admm_dual_plane(
    tm_system,                 # a CALIBRATED TMMeasurementSystem
    train_input_modes,         # Z_in, shape (N_modes, Batch)
    train_intensity_targets,   # (Batch, 2, H, W); 0 = image, 1 = Fourier
    num_epochs=..., planes='both'|'image'|'fourier',
    init_method='default_beta_prior',
    resync_interval=...,
)
# -> {'epoch_losses': [...], 'epoch_times': [...]}
```

`default_beta_prior` leaves the TM that `calibrate()` just wrote: `diag(e^{iβ})` plus complex Gaussian noise with σ = 0.01.

## Reproducing each figure

Figure scripts rebase paths on the repo root and can be run from anywhere.

### Wavefront control, cross-plane focusing, vectorial control

```bash
python paper/make_control_figures.py
```

Reads PNGs from `paper/image_results/` (captured by `lab/run_wavefront_control.py`) and writes `paper/figures/tm_wavefront_control.png`, `tm_cross_plane_focus.png`, `tm_vectorial_control.png`. Panel (c) of the control figure uses the lab one-hot mode-excitation mask (`projected_full_phase_mask_uint8_one_hot_modes.npy`), not the offline runner.

### Focus versus reconstruction time

```bash
python paper/make_focus_vs_time_figure.py
```

Reads measured CSVs from `results/lab_replay/` and offline epoch tables from `results/offline/{modal,raf21,admm}/`. Writes `paper/figures/lab_focus_vs_accumulated_time.png` and `lab_focus_vs_accumulated_time_fourier_plane.png`.

## The algorithms

Detection is polarization-summed, `I = |E_x|² + |E_y|²`, for two planes at once (index 0 = image, index 1 = Fourier).

**Algorithm 1 — `optimize_tm_admm_dual_plane` in `TM_retrieval_algos.py`.** Autograd-free modal ADMM. Closed-form TM update, then a joint polarization-preserving amplitude projection, then a leaky dual update `γ_t = 0.9^{⌊t/15⌋}`. Over-relaxation `ρ = 0` for the first 45 iterations, then `0.1`. The amplitude projection acts jointly on the two polarization components and therefore leaves their relative phase to the vector mode profiles, which is what allows a vectorial TM to be retrieved with no polarizer in the detection path.

**Algorithm 2 — `optimize_tm_slm` in `TM_retrieval_algos.py`.** Differentiable forward model with aberration corrections. Adam jointly fits the TM and the Zernike / ZoomFFT nuisance parameters by minimizing the PCC loss.

The `planes` argument selects which measurement plane enters the loss. A TM retrieved from one plane is deliberately evaluated by focusing in the complementary plane (`_focus_eval_plane_for_pcc_planes` in `run_modal_tm_retrieval.py`). That cross-plane transfer is a claim of the paper, not a bug. `demo.py --quick` shows a miniature of the same idea on synthetic data: after an image-only fit, PCC on the seen image plane and on the unseen Fourier plane are printed side by side.

Everything else that used to live in `TM_retrieval_algos.py` (MSE/PCC line-search, L-BFGS, analytic, 4-step internal-reference, lifted ALS, spectral init) is not in the published tree.

## Code map

| Module | Role |
|---|---|
| `mode_solver.py` | Step-index LP dispersion solve (`find_guided_modes`). |
| `utils.py` | Mode-field rendering and cache, `ZoomFFT2D_Optimizable`, `ZernikeAberrationLayer`, H5 loaders, and `sample_tm_perturbed_mmf` (random-matrix TM sampler used only by the `lab_data=False` simulation path). |
| `TM_retrieve.py` | `TMMeasurementSystem`: forward chain `slm_to_input_modes` → `input_modes_to_output_modes` → `output_modes_to_intensities` → `forward_model_slm`, plus `calibrate()`. |
| `TM_retrieval_algos.py` | The two published solvers. |
| `metrics_losses.py` | `PearsonCorrelationLoss` (`1 - PCC`, the scale-invariant intensity objective) and the focus enhancement factor (target window over the fiber core / NA disk; `_focus_metric_from_intensity_map`, `enhancement_factor`). |
| `motor_pos_utils.py` | Run labels and deterministic subset sampling. |

`TMMeasurementSystem` is the forward model only. Exploratory plotting, evaluation wrappers, `main()`, and `enforce_symmetric_tm` were removed for the release, so a reader comparing against an earlier draft is not looking at a different optical model — those paths were unused in the paper.

Forward chain, in order:

1. `slm_to_input_modes` — SLM phase (and optional amplitude mask) to input guided-mode coefficients `Z_in`.
2. `input_modes_to_output_modes` — multiply by the TM.
3. `output_modes_to_intensities` — after `calibrate()`, one matmul per plane per polarization, then `I = |E_x|² + |E_y|²`.
4. `forward_model_slm` — the composition of (1)–(3).

Subset benchmarks sample probe indices with `motor_pos_utils.sample_random_subset_indices(n, size, trial, seed)`, deterministic in `(seed, trial, size)`. The join key across H5 groups, CSVs and JSON is `{method}_subset_{size}_trial_{idx}`. See `results/README.md`.

## Gotchas

- `calibrate()` resets the TM via `reinitialize_optimized_tm()`. Snapshot `optimized_tm` first if you need to keep it. Re-call `calibrate()` if any output-path parameter changes.
- Mode fields are cached by a parameter hash. Paper scripts write `mode_fields_<hash>.pt` in the **current working directory**; run them from the repo root or the cache is silently recomputed. `demo.py` writes under `data/mode_fields/`. Changing core radius, index, wavelength, or grid produces a new multi-hundred-MB file.
- `meas_scale` prescales measurements for float32 range and rescales the recovered TM by `sqrt(meas_scale)`. Reported metrics are scale- and global-phase-invariant.
- Intensity-only retrieval cannot determine the global phase. The polarization basis is recovered only up to one global rotation angle θ.

Fiber numbers used in the paper scripts (do not invent others): `a = 25e-6`, `n1 = 1.457`, `NA = 0.22`, `λ = 632.8e-9`, grid 128×128, `M = 1508`.

## Hardware phase

Acquisition and validation need the bench, `pypylon`, and `create_refs`. See `lab/README.md`.

- `lab/acquire_modal_dataset.py` writes `data/measurement_motor_pcc.h5` (both groups, motorized displacement between them).
- `lab/acquire_baseline_dataset.py` writes `data/measurement_raf.h5`.
- `lab/run_wavefront_control.py` writes the camera PNGs in `paper/image_results/` and the one-hot mode-excitation mask used in Fig. tm_control panel (c).
- `lab/replay_modal_masks.py` / `lab/replay_baseline_masks.py` write the measured CSVs in `results/lab_replay/`.

The optical layout, as used by the code: SLM on one polarization, two cameras on image and Fourier planes, 128×128 frames, 1 m MMF with M = 1508 modes at 632.8 nm.

## Tests

```bash
python -m pytest tests -q
```

`tests/test_smoke.py` imports every published module, AST-parses `lab/*.py`, and runs a short synthetic ADMM retrieval asserting that the loss decreases and that both planes reach a positive correlation. It reuses `demo.run_demo`. On this machine the full `tests/` suite finished in a few seconds of CPU time.

## License, citation, contact

Licensing is pending institutional approval (`LICENSE` is a placeholder). **TODO:** replace it after university approval.

Cite the manuscript (authors: Dvir Marsh, Lior Fridman, Shie Mannor, Guy Bartal). DOI/year in `CITATION.cff` are placeholders.

Contact: the corresponding author on the manuscript.
