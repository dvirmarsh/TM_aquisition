"""
Stage B (revised, 15 Nov 2026), step 1 (offline, no hardware): check the new dataset and the new
polarization-blind TM.

    python stageB_step1_check.py

Reads measurement_focus.h5 and tm_focus_outputs/tm_system_checkpoint.pt (never writes them).
Writes tm_focus_outputs/stageB_offline/step1_check.json.

* Speckle contrast = std / mean over the support pixels of one frame (fiber core in the image plane,
  NA disk in the Fourier plane, r <= 58.1 px), after dividing out the mean image of all masks,
  averaged over the masks.
* Held-out correlation = Pearson correlation between the predicted and the measured intensity frame
  of a held-out probe mask (302 masks, split 0.95 with seed 42 as in the pipeline), full frame,
  averaged over the masks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import retrieve_tm_camera_weights as rcw       # noqa: E402  (stubs create_refs: no hardware library is loaded)
import stageA_offline_analysis as sa           # noqa: E402

OUT_DIR = TM_LIB / "tm_focus_outputs" / "stageB_offline"
# Values of the dataset of 12 Nov, 12:34 (Stage A report); that file has since been overwritten.
OLD = {"fourier": {"contrast": 0.822, "contrast_raw": 0.926, "mean_level": 427.1},
       "image": {"contrast": 1.033, "contrast_raw": 1.064, "mean_level": 36.1},
       "held_out_blind": {"image": 0.792, "fourier": 0.829}}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    support = sa.disk(sa.SUPPORT_RADIUS)
    with h5py.File(TM_LIB / "measurement_focus.h5", "r") as h5:
        g = h5["experiment_0"]
        four = np.asarray(g["transmission_fourier"][:], dtype=np.float32)
        im = np.asarray(g["transmission_image"][:], dtype=np.float32)
        n_masks = int(g["PhaseMask"].shape[0])
    res = {"n_masks": n_masks}
    for name, frames in (("fourier", four), ("image", im)):
        c = sa.speckle_contrast(frames, support)
        c["max_level"] = float(frames.max())
        c["median_outside_support"] = float(np.median(frames[:, ~support]))
        # drift of the total power over the acquisition: mean level of the first and last 500 masks
        c["mean_level_first_500"] = float(frames[:500][:, support].mean())
        c["mean_level_last_500"] = float(frames[-500:][:, support].mean())
        res[name] = c
        print(f"{name:7s} plane: speckle contrast {c['contrast_envelope_removed']:.3f} (12 Nov: {OLD[name]['contrast']:.3f}); "
              f"raw {c['contrast_raw']:.3f} (12 Nov: {OLD[name]['contrast_raw']:.3f}); mean level in the support {c['mean_level_in_support']:.1f} counts "
              f"(12 Nov: {OLD[name]['mean_level']:.1f}); first / last 500 masks {c['mean_level_first_500']:.1f} / {c['mean_level_last_500']:.1f}; "
              f"maximum {c['max_level']:.0f}; saturated pixels {100 * c['saturated_pixel_fraction']:.3f} %; "
              f"median outside the support {c['median_outside_support']:.1f}")

    cfg = rcw.rtf.PipelineConfig()
    pipeline = rcw.rtf.TMFocusPipeline(cfg)                      # geometry files only; setup() is never called
    _train, test_data = rcw.load_data(cfg, pipeline)
    chk = rcw.check_b0(pipeline, test_data)                      # also: copied code == original forward model
    ho = chk["held_out_current_checkpoint"]
    res["blind_tm"] = chk
    print(f"new polarization-blind TM, held-out correlation: image {ho['image']['pcc_full_frame']:.3f} (12 Nov: {OLD['held_out_blind']['image']:.3f}), "
          f"Fourier {ho['fourier']['pcc_full_frame']:.3f} (12 Nov: {OLD['held_out_blind']['fourier']:.3f}); "
          f"inside the support: {ho['image']['pcc_in_support']:.3f}, {ho['fourier']['pcc_in_support']:.3f}")
    print(f"copied code against the checkpoint's forward model: largest difference {chk['max_abs_diff_intensity_b0']:.3g}")
    json.dump(res, open(OUT_DIR / "step1_check.json", "w"), indent=1)
    print(f"Saved {OUT_DIR / 'step1_check.json'}")


if __name__ == "__main__":
    main()
