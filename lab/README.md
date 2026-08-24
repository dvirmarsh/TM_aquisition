# Lab scripts (hardware only)

These scripts drive the bench. They cannot run for a reader: they import `pypylon` (Basler camera SDK) and a `create_refs` module that lives in a **private parent repository**, not in this checkout. They are included so a porter can see the acquire → retrieve → validate loop, not as runnable code.

Each file puts this repository and its parent on `sys.path`:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))            # this repository's core modules
sys.path.insert(0, str(REPO_ROOT.parent))     # the private lab repo providing create_refs
```

Missing hardware must fail loudly. There are no stubs for `pypylon` or `create_refs`.

## What each script produces

| Script | Artifact |
|---|---|
| `acquire_modal_dataset.py` | `data/measurement_motor_pcc.h5` (groups `initial_calibration` and `post_motor`, with a motorized fiber displacement between them) |
| `acquire_baseline_dataset.py` | `data/measurement_raf.h5` (32×32 SLM macro-pixels, kron-upsampled by 20 to the physical aperture; no motor; no calibration group) |
| `run_wavefront_control.py` | camera PNGs in `paper/image_results/` behind the wavefront-control, cross-plane and vectorial figures; also `projected_full_phase_mask_uint8_one_hot_modes.npy` (Fig. tm_control panel c) |
| `replay_modal_masks.py` | measured `results/lab_replay/modal_replay_epoch_table.csv` (and the Fourier-plane companion CSV) |
| `replay_baseline_masks.py` | measured `results/lab_replay/raf21_lab_replay_epoch_table.csv` and `admm_lab_replay_epoch_table.csv` |

## Optical setup (as used by the code)

SLM modulating one polarization component, two cameras exposing simultaneously on the image and Fourier planes, 128×128 recorded frames, a 1 m MMF supporting M = 1508 modes at λ = 632.8 nm (`a = 25e-6`, `n1 = 1.457`, `NA = 0.22`). Camera serial numbers and exposure times are fields on `PipelineConfig` in `run_wavefront_control.py`.

## `create_refs` interface, from call sites only

The following names are imported. Signatures below are what the **callers** pass; return-value contracts are the methods those callers then invoke. Nothing else is claimed.

**`create_SLM()`** — called with no arguments. The first element of the returned tuple is the SLM object when `PipelineConfig.slm_orig_lib` is True. That object is used as:

- `Write_image(phase_mask.flatten().ctypes.data_as(POINTER(c_ubyte)), c_uint(1))`
- `Delete_SDK()` on teardown

**`create_SLMpy()`** — called with no arguments. The first element of the returned 3-tuple is the SLM object when `slm_orig_lib` is False. Used as:

- `updateArray(phase_mask)`
- `close()` on teardown

**`create_cmeras(serial_numbers, exposure_lists)`** — first argument is `list(self.cfg.cameras_sn)` (four Basler serial strings). Second argument is a list of four single-element exposure-time lists (image, Fourier, then the same pair repeated). Returns a sequence of camera objects. Callers use:

- index 0 as the transmission (image-plane) camera in replay
- `cam.Close()` on teardown
- `set_camera_pixel_format(cam, pixel_format)` and `set_camera_pixel_format(cam, pixel_format, exposure_time=...)`
- `get_camera_exposure_minimum(cam)`
- `capture_hdr_images(cam, [exposure])` is a method on `TMFocusPipeline`, not on `create_refs`

**`connect_motor()`** — called with no arguments. Returned object:

- `move_to(position_mm, blocking=True)` (also used by `motor_pos_utils.home_motor`)
- `home(blocking=True)` or `home()` (tried in that order by `home_motor`)

**`get_camera_exposure_minimum(camera)`** — one positional argument, the camera object. Return value is used as an exposure time in milliseconds (or the unit `capture_hdr_images` expects).

**`set_camera_pixel_format(camera, pixel_format, exposure_time=...)`** — `pixel_format` is a string (`"Mono12"` or `"Mono8"`). `exposure_time` is passed as a keyword at the acquire-modal setup site and omitted elsewhere.

Open questions (not determined from these call sites): the remaining tuple elements of `create_SLM` / `create_SLMpy`; the exact camera class; whether `create_cmeras` always returns four cameras; units of exposure; the full motor API.
