"""
PBR bias check (15 Nov 2026), image-plane camera only. The rotation stage is not used.

Purpose: the PBR of a single focus is the peak pixel over the mean background of ONE frame at the shortest exposure,
where the background is 1 to 2 counts and about half of its pixels read 0. This script measures how much that biases
the PBR, with the settings of the standard single-focus capture of run_tm_focus_optimization.py (Mono12, exposure 24,
crop + bilinear resize to 128 x 128, flip).

    python run_pbr_bias_check.py [--allow-new-dataset] [--min-freshness 0.9]

Steps
 1. Freshness of the current standard TM: 8 held-out probe masks are displayed and recorded at the dataset exposure;
    correlation with the dataset frames of the same masks (inside r <= 50.1 px), measured a second time if below the
    limit. Below the limit twice: stop, or with --allow-new-dataset record a new dataset and retrieve a new TM with
    the code of the standard pipeline into NEW files (measurement_focus.h5 and tm_system_checkpoint.pt are not touched).
 2. Five single-focus targets in the image plane, sampled and optimized as in the standard run
    (sample_target_pixel_near_center, optimize_slm_pattern_with_tm, 150 iterations, 1 x 1 target window).
 3. Per target: the pattern is displayed and five frames are recorded at each exposure E0, 4 E0, 16 E0, 64 E0
    (E0 = 24, or the camera minimum if that is larger; also at 20 if the camera accepts it). One warm-up frame is
    discarded after every change of pattern or exposure.
 4. Per target, PBR = peak pixel / mean of the fiber core outside the 3 x 3 window, three ways:
      (i)   from one E0 frame (mean and spread over the five frames),
      (ii)  peak from E0, background from the longest exposure whose background has fewer than 1 % zero pixels,
            without saturated pixels and without a 7 x 7 window around the peak, scaled by the exposure ratio,
      (iii) as (ii) with the next shorter exposure.
    Exposure linearity on the camera pixels between 50 and 3000 counts.
 5. Pixel format, BlackLevel and Gain are read from the camera before and after the frames. The pipeline's teardown
    puts the cameras back to Mono8 at the minimum exposure, as after every standard run.

Output: tm_focus_outputs/pbr_bias_check_<date>_<time>/ (summary JSON, 128 x 128 frames, raw camera crops, SLM patterns).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch

TM_LIB = Path(__file__).resolve().parent
for _p in (TM_LIB.parent, TM_LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SAT = 4095
E0_STANDARD = 24.0                     # run_tm_focus_optimization.py, line 1535
E_REPOSITORY = 20.0                    # TM_acquisition/lab/run_wavefront_control.py, line 965
FACTORS = (1, 4, 16, 64)
N_FRAMES = 5
N_TARGETS = 5
N_FRESH = 8
FIT_RADIUS = 50.1
MAX_BETWEEN_MASKS = 0.5                # frames of different probe masks are uncorrelated when the SLM works (about 0 on 15 Nov)
TARGET_SEED = 20261115
CROP_NAME = "transmission_image"


def disk(n: int, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:n, :n]
    return (yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2 <= radius ** 2


def held_out_indices(n_total: int, split: float = 0.95, seed: int = 42) -> np.ndarray:
    """Indices of the held-out probe masks of the pipeline (utils.load_real_mmf_data)."""
    from sklearn.model_selection import train_test_split
    _train, test = train_test_split(np.arange(n_total), test_size=1 - split, random_state=seed)
    return np.sort(test)


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------------------
class ImageCamera:
    """SLM and the image-plane camera through TMFocusPipeline, with one discarded frame after every change."""

    def __init__(self, pipeline):
        self.p = pipeline
        self.cam = pipeline.cameras[0]
        self.half = int(pipeline.n_pix_map[CROP_NAME])
        self.yc, self.xc = int(pipeline.y_centers[CROP_NAME]), int(pipeline.x_centers[CROP_NAME])
        self._dirty, self._last_exposure, self.n_warmup = True, None, 0

    def show(self, mask_small_u8: np.ndarray) -> None:
        self.p.project_phase_mask_to_slm(self.p.pack_phase_to_full_slm(mask_small_u8))
        self._dirty = True

    def grab(self, exposure: float) -> np.ndarray:
        """Full sensor frame (uint16 counts)."""
        e = float(exposure)
        if self._dirty or e != self._last_exposure:
            self.p.capture_hdr_images(self.cam, [e])                       # warm-up frame, discarded
            self.n_warmup += 1
        self._dirty, self._last_exposure = False, e
        return np.squeeze(np.array(self.p.capture_hdr_images(self.cam, [e])))

    def raw_crop(self, raw: np.ndarray) -> np.ndarray:
        return raw[self.yc - self.half:self.yc + self.half, self.xc - self.half:self.xc + self.half]

    def as_dataset(self, raw: np.ndarray) -> np.ndarray:
        """Frame as stored in the dataset: crop + resize, not flipped (save_image_to_h5)."""
        return np.round(self.p.crop_and_resize_image(raw, CROP_NAME)).astype(np.float32)

    def as_focus_frame(self, raw: np.ndarray) -> np.ndarray:
        """Frame as evaluated by the standard single-focus step: crop + resize + flip (lines 1537-1539)."""
        return np.flip(self.p.crop_and_resize_image(raw, CROP_NAME).astype(np.float32)).copy()


def camera_state(cam) -> Dict:
    """Read-only: identity and the settings that decide the dark level."""
    out: Dict = {}
    try:
        info = cam.GetDeviceInfo()
        for name, fn in (("serial", "GetSerialNumber"), ("model", "GetModelName"), ("vendor", "GetVendorName"), ("device_class", "GetDeviceClass")):
            try:
                out[name] = str(getattr(info, fn)())
            except Exception as exc:                                      # noqa: BLE001
                out[name] = f"not available ({type(exc).__name__})"
    except Exception as exc:                                              # noqa: BLE001
        out["device_info"] = f"not available ({type(exc).__name__})"
    nodes = ("PixelFormat", "BlackLevel", "BlackLevelSelector", "Gain", "GainSelector", "GainAuto", "ExposureAuto", "ExposureMode",
             "ExposureTime", "Gamma", "DigitalShift", "BinningHorizontal", "BinningVertical", "BinningHorizontalMode", "BinningVerticalMode",
             "Width", "Height", "OffsetX", "OffsetY", "SensorWidth", "SensorHeight", "ReverseX", "ReverseY", "TriggerMode", "TriggerSource",
             "BslUSBSpeedMode", "DeviceLinkSpeed", "SensorReadoutMode", "ShutterMode", "DeviceModelName", "DeviceFirmwareVersion",
             "BslContrastMode", "BslBrightness", "BslContrast", "PgiMode", "DefectPixelCorrectionMode")
    for node in nodes:
        try:
            n = getattr(cam, node)
            v = n.GetValue()
            out[node] = v if isinstance(v, (int, float, bool, str)) else str(v)
            if node in ("BlackLevel", "Gain", "ExposureTime"):
                for extra, fn in (("min", "GetMin"), ("max", "GetMax"), ("unit", "GetUnit")):
                    try:
                        out[f"{node}_{extra}"] = getattr(n, fn)()
                    except Exception:                                     # noqa: BLE001
                        pass
        except Exception as exc:                                          # noqa: BLE001
            out[node] = f"not available ({type(exc).__name__})"
    return out


# ---------------------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------------------
def freshness(rig: ImageCamera, masks: np.ndarray, dataset_frames: np.ndarray, exposure: float) -> Dict:
    fit = disk(masks.shape[-1], FIT_RADIUS)
    frames = []
    for k in range(len(masks)):
        rig.show(masks[k])
        frames.append(rig.as_dataset(rig.grab(exposure)))
    frames = np.stack(frames)
    same = [pcc(frames[k][fit], dataset_frames[k][fit]) for k in range(len(masks))]
    full = [pcc(frames[k], dataset_frames[k]) for k in range(len(masks))]
    prev = [pcc(frames[k][fit], dataset_frames[k - 1][fit]) for k in range(1, len(masks))]
    between = [pcc(frames[i][fit], frames[j][fit]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
    return {"pcc_fit_region": same, "mean": float(np.mean(same)), "pcc_full_frame_mean": float(np.mean(full)),
            "mean_against_the_previous_mask": float(np.mean(prev)), "exposure": float(exposure),
            "correlation_between_frames_of_different_masks": float(np.mean(between)),
            "level_now_over_dataset": float(frames[:, fit].mean() / max(dataset_frames[:, fit].mean(), 1e-9)),
            "saturated_pixels_per_frame": float((frames >= SAT).sum() / len(frames)), "frames": frames}


def new_dataset_and_tm(pipeline, rtf, out_dir: Path, stamp: str):
    """A new dataset and a new TM with the code of the standard pipeline, written to NEW files."""
    cfg = pipeline.cfg
    cfg.output_h5_path = str(TM_LIB / f"measurement_focus_pbr_check_{stamp}.h5")
    cfg.tm_checkpoint_path = str(out_dir / "tm_system_checkpoint_pbr_check.pt")
    assert not Path(cfg.output_h5_path).exists() and not Path(cfg.tm_checkpoint_path).exists()
    free_gb = shutil.disk_usage(str(TM_LIB)).free / 1e9
    if free_gb < 5.0:
        raise RuntimeError(f"Only {free_gb:.1f} GB free; a new dataset and TM need about 1 GB.")
    t0 = time.time()
    h5_path = pipeline.sample_transmission_data()                          # as Step 1/6 of TMFocusPipeline.run
    t_acq = (time.time() - t0) / 60.0
    et_ratio = cfg.transmission_image_et[0] / cfg.transmission_fourier_et[0]
    bs_split = 0.47736                                                     # as in TMFocusPipeline.run
    tran_im_factor = pipeline.magnification ** 2 * bs_split / (1 - bs_split) * et_ratio
    train_data, test_data = rtf.load_real_mmf_data(h5_file_path=h5_path, experiment_group=cfg.output_group,
                                                   train_test_split_ratio=cfg.train_test_split_ratio, random_seed=cfg.random_seed,
                                                   im_factor=tran_im_factor)
    tm = pipeline.initialize_tm_system()
    for d in (train_data, test_data):
        d["slm_patterns"] = d["slm_patterns"].to(tm.device)
        d["intensity_targets"] = d["intensity_targets"].to(tm.device)
    t1 = time.time()
    history = rtf.optimize_tm_slm(tm_system=tm, slm_patterns=train_data["slm_patterns"], intensity_targets=train_data["intensity_targets"],
                                  num_epochs=cfg.tm_num_epochs, learning_rate=cfg.tm_learning_rate, batch_size=cfg.tm_batch_size,
                                  planes=cfg.optimize_planes, optimize_aberrations=cfg.optimize_aberrations, test_data=test_data,
                                  use_exposure_normalization=cfg.use_exposure_normalization)
    torch.save(tm, cfg.tm_checkpoint_path)
    info = {"dataset": cfg.output_h5_path, "checkpoint": cfg.tm_checkpoint_path, "acquisition_minutes": t_acq,
            "retrieval_minutes": (time.time() - t1) / 60.0, "epochs": int(cfg.tm_num_epochs),
            "final_train_loss": float(history["epoch_losses"][-1]),
            "final_test_loss": float(history["test_losses"][-1]) if history.get("test_losses") else None}
    del train_data, test_data
    torch.cuda.empty_cache()
    return tm, info


def design_targets(pipeline, tm, out_dir: Path) -> List[Dict]:
    cfg = pipeline.cfg
    n = int(cfg.n_out)
    np.random.seed(TARGET_SEED)
    targets: List[Tuple[int, int]] = []
    while len(targets) < N_TARGETS:
        t = pipeline.sample_target_pixel_near_center((n, n), cfg.target_max_offset_from_center_px * 0.3)   # as line 1508
        if t not in targets:
            targets.append(t)
    out = []
    for k, t in enumerate(targets):
        torch.manual_seed(TARGET_SEED + k)
        phase, history = pipeline.optimize_slm_pattern_with_tm(tm_system=tm, slm_shape=(n, n), num_iterations=cfg.slm_opt_iters, lr=cfg.slm_opt_lr,
                                                               center_radius=cfg.focus_center_radius_px, target_plane="image", target_yx=t)
        phase_np = phase.detach().cpu().numpy()
        u8 = pipeline.phase_to_uint8(phase_np)
        with torch.no_grad():
            pred = tm.forward_model_slm(torch.as_tensor(u8.astype(np.float32) * (2 * np.pi / 256.0), device=tm.device)[None])[0, 0].cpu().numpy()
        np.save(out_dir / f"target_{k + 1}_phase_uint8.npy", u8)
        np.save(out_dir / f"target_{k + 1}_phase_rad.npy", phase_np.astype(np.float32))
        out.append({"index": k + 1, "target_yx": [int(t[0]), int(t[1])], "phase_uint8": u8, "predicted_image": pred,
                    "focus_metric_last_iteration": float(history[-1])})
        print(f"  target {k + 1}: (y, x) = {t}, predicted focus metric {history[-1]:.0f}")
    return out


def record_target(rig: ImageCamera, u8: np.ndarray, exposures: Sequence[float]) -> Dict:
    rig.show(u8)
    raw_crops, frames = {}, {}
    for e in exposures:
        raws = [rig.grab(e) for _ in range(N_FRAMES)]
        raw_crops[float(e)] = np.stack([rig.raw_crop(r) for r in raws]).astype(np.uint16)
        frames[float(e)] = np.stack([rig.as_focus_frame(r) for r in raws])
    return {"raw_crops": raw_crops, "frames": frames}


# ---------------------------------------------------------------------------------------
# analysis (offline; also used on saved frames)
# ---------------------------------------------------------------------------------------
def analyse_target(frames: Dict[float, np.ndarray], raw_crops: Dict[float, np.ndarray], support: np.ndarray, target_yx: Tuple[int, int],
                   e0: float, pbr_fn, search_radius: int = 3) -> Dict:
    """frames[e]: (N_FRAMES, n, n) focus frames; raw_crops[e]: (N_FRAMES, c, c) uint16. pbr_fn = rtf.peak_to_background_ratio."""
    n = support.shape[0]
    yy, xx = np.mgrid[:n, :n]
    exposures = sorted(e for e in frames if e >= e0)
    f0 = frames[e0]
    pbr_single, peaks = [], []
    for f in f0:
        v, (py, px) = pbr_fn(f, support, target_yx=target_yx, search_radius=search_radius, half_width=1)
        pbr_single.append(v)
        peaks.append((py, px))
    _v, (py, px) = pbr_fn(f0.mean(axis=0), support, target_yx=target_yx, search_radius=search_radius, half_width=1)
    win3 = (np.abs(yy - py) <= 1) & (np.abs(xx - px) <= 1)
    win7 = (np.abs(yy - py) <= 3) & (np.abs(xx - px) <= 3)
    peak0 = float(np.mean([f[p] for f, p in zip(f0, peaks)]))
    bg3_0 = support & ~win3
    res: Dict = {"target_yx": [int(target_yx[0]), int(target_yx[1])], "peak_yx": [int(py), int(px)], "e0": float(e0),
                 "peak_e0_counts_mean": peak0, "peak_e0_counts_frames": [float(f[p]) for f, p in zip(f0, peaks)],
                 "peak_position_per_frame": [[int(a), int(b)] for a, b in peaks],
                 "pbr_i_frames": [float(v) for v in pbr_single], "pbr_i": float(np.mean(pbr_single)), "pbr_i_std": float(np.std(pbr_single)),
                 "pbr_i_first_frame": float(pbr_single[0]),
                 "background_e0_mean_counts": float(np.mean([f[bg3_0].mean() for f in f0])),
                 "background_e0_zero_fraction": float(np.mean([(f[bg3_0] == 0).mean() for f in f0])),
                 "background_e0_fraction_at_1_count": float(np.mean([(f[bg3_0] == 1).mean() for f in f0])),
                 "per_exposure": {}}
    for e in exposures:
        fr = frames[e]
        sat_raw = (raw_crops[e] >= SAT).any(axis=0).astype(np.float32)
        sat_grid = np.flip(cv2.resize(sat_raw, (n, n), interpolation=cv2.INTER_LINEAR)) > 0
        sat_grid |= (fr >= SAT).any(axis=0)
        mask7 = support & ~win7 & ~sat_grid
        mask3 = support & ~win3 & ~sat_grid
        res["per_exposure"][f"{e:g}"] = {
            "exposure": float(e), "ratio_to_e0": float(e / e0),
            "background_7x7_mean_counts": float(fr[:, mask7].mean()), "background_7x7_scaled_to_e0": float(fr[:, mask7].mean() * e0 / e),
            "background_7x7_zero_fraction": float((fr[:, mask7] == 0).mean()),
            "background_3x3_mean_counts": float(fr[:, mask3].mean()), "background_3x3_scaled_to_e0": float(fr[:, mask3].mean() * e0 / e),
            "background_3x3_zero_fraction": float((fr[:, mask3] == 0).mean()),
            "saturated_grid_pixels_excluded": int((support & sat_grid).sum()), "saturated_camera_pixels": int(sat_raw.sum()),
            "peak_pixel_mean_counts": float(fr[:, py, px].mean()), "largest_count": float(fr.max())}
    ok = [e for e in exposures if res["per_exposure"][f"{e:g}"]["background_7x7_zero_fraction"] < 0.01]
    res["exposures_with_fewer_than_1_percent_zeros"] = [float(e) for e in ok]
    res["criterion_fewer_than_1_percent_zeros_met"] = bool(ok)
    if len(exposures) > 1:
        e_long = max(ok) if ok else max(exposures)                         # if no exposure meets the criterion: the longest, flagged
        shorter = [e for e in exposures if e < e_long]
        e_next = max(shorter) if shorter else None
        b_long = res["per_exposure"][f"{e_long:g}"]["background_7x7_scaled_to_e0"]
        res.update({"exposure_ii": float(e_long), "pbr_ii": peak0 / b_long, "ratio_i_over_ii": res["pbr_i"] / (peak0 / b_long)})
        if e_next is not None:
            b_next = res["per_exposure"][f"{e_next:g}"]["background_7x7_scaled_to_e0"]
            res.update({"exposure_iii": float(e_next), "pbr_iii": peak0 / b_next, "ratio_i_over_iii": res["pbr_i"] / (peak0 / b_next)})
        b7_0 = res["per_exposure"][f"{e0:g}"]["background_7x7_scaled_to_e0"]
        b3_long = res["per_exposure"][f"{e_long:g}"]["background_3x3_scaled_to_e0"]
        res.update({"pbr_i_with_7x7_exclusion": peak0 / b7_0, "ratio_i7_over_ii": (peak0 / b7_0) / (peak0 / b_long),
                    "implied_offset_counts_per_pixel": b_long - b7_0,
                    "pbr_ii_with_3x3_exclusion": peak0 / b3_long, "ratio_i_over_ii_3x3": res["pbr_i"] / (peak0 / b3_long)})
        res["background_ii_zero_fraction"] = res["per_exposure"][f"{e_long:g}"]["background_7x7_zero_fraction"]
        # The paper's background (support outside the 3 x 3 window) without the dark-level bias: the ring between the
        # 3 x 3 and the 7 x 7 window is bright and is taken from the E0 frames; the rest from the long exposure.
        ring = support & win7 & ~win3
        n_out7 = int((support & ~win7).sum())
        b3_true = (float(f0.mean(axis=0)[ring].sum()) + b_long * n_out7) / (int(ring.sum()) + n_out7)
        res.update({"ring_3x3_to_7x7_mean_counts_e0": float(f0.mean(axis=0)[ring].mean()), "ring_pixels": int(ring.sum()),
                    "background_3x3_definition_unbiased_counts": b3_true, "pbr_same_definition_unbiased": peak0 / b3_true,
                    "ratio_i_over_same_definition": res["pbr_i"] / (peak0 / b3_true),
                    "share_of_ring_in_background_e0": float(f0.mean(axis=0)[ring].sum() / max(f0.mean(axis=0)[bg3_0].sum(), 1e-9))})
        if not ok:
            res["note"] = "no exposure has fewer than 1 % zero pixels in the background; (ii) uses the longest exposure"
    # exposure linearity on the camera pixels (mean of the frames), 50 .. 3000 counts at both exposures, no saturated frame
    lin = {}
    means = {e: raw_crops[e].astype(np.float64).mean(axis=0) for e in exposures}
    unsat = {e: ~(raw_crops[e] >= SAT).any(axis=0) for e in exposures}
    for i, ea in enumerate(exposures):
        for eb in exposures[i + 1:]:
            a, b = means[ea], means[eb]
            sel = (a >= 50) & (a <= 3000) & (b >= 50) & (b <= 3000) & unsat[ea] & unsat[eb]
            if sel.sum() < 20:
                lin[f"{ea:g}_to_{eb:g}"] = {"n_pixels": int(sel.sum())}
                continue
            x = a[sel] * (eb / ea)
            slope, offset = np.polyfit(x, b[sel], 1)
            lin[f"{ea:g}_to_{eb:g}"] = {"n_pixels": int(sel.sum()), "ratio_of_sums_over_exposure_ratio": float(b[sel].sum() / x.sum()),
                                        "fit_slope": float(slope), "fit_offset_counts_at_longer_exposure": float(offset),
                                        "implied_dark_offset_counts": float(-offset / (eb / ea - 1.0))}
    res["linearity"] = lin
    return res


def summarise(per_target: List[Dict]) -> Dict:
    out = {}
    for key in ("pbr_i", "pbr_ii", "pbr_iii", "ratio_i_over_ii", "ratio_i_over_iii", "ratio_i7_over_ii", "ratio_i_over_ii_3x3",
                "pbr_same_definition_unbiased", "ratio_i_over_same_definition", "share_of_ring_in_background_e0", "ring_3x3_to_7x7_mean_counts_e0",
                "background_e0_mean_counts", "background_e0_zero_fraction", "implied_offset_counts_per_pixel"):
        v = np.array([t[key] for t in per_target if key in t], dtype=np.float64)
        if len(v):
            out[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    return out


def print_report(res: Dict) -> None:
    print("\ntarget  peak(E0)  bg(E0)  zeros(E0)   PBR (i)        PBR (ii)  PBR (iii)  (i)/(ii)  (i)/(iii)  exposures of (ii), (iii)")
    for t in res["targets"]:
        a = t["analysis"]
        print(f"  {t['index']}  {str(tuple(a['peak_yx'])):>9} {a['peak_e0_counts_mean']:7.0f} {a['background_e0_mean_counts']:7.3f} {a['background_e0_zero_fraction']:8.3f} "
              f"{a['pbr_i']:8.1f} +- {a['pbr_i_std']:5.1f} {a.get('pbr_ii', float('nan')):9.1f} {a.get('pbr_iii', float('nan')):9.1f} "
              f"{a.get('ratio_i_over_ii', float('nan')):8.3f} {a.get('ratio_i_over_iii', float('nan')):9.3f}    {a.get('exposure_ii', '-')}, {a.get('exposure_iii', '-')}")
    s = res["summary"]
    if "ratio_i_over_ii" in s:
        print(f"ratio (i)/(ii): mean {s['ratio_i_over_ii']['mean']:.3f}, spread (std over the targets) {s['ratio_i_over_ii']['std']:.3f}, "
              f"range {s['ratio_i_over_ii']['min']:.3f} to {s['ratio_i_over_ii']['max']:.3f}")
    if "ratio_i7_over_ii" in s:
        print(f"the same with the 7 x 7 window excluded at E0 as well: mean {s['ratio_i7_over_ii']['mean']:.3f}, std {s['ratio_i7_over_ii']['std']:.3f}")
    if "ratio_i_over_same_definition" in s:
        print(f"(i) over the PBR with the same background definition (3 x 3) and the long-exposure background: "
              f"mean {s['ratio_i_over_same_definition']['mean']:.3f}, std {s['ratio_i_over_same_definition']['std']:.3f}, "
              f"range {s['ratio_i_over_same_definition']['min']:.3f} to {s['ratio_i_over_same_definition']['max']:.3f}; "
              f"the ring between the two windows holds {100 * s['share_of_ring_in_background_e0']['mean']:.0f} % of the E0 background counts")


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------
def run_check(pipeline, rtf, out_dir: Path, allow_new_dataset: bool, min_freshness: float = 0.9, device: str = "cuda") -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = pipeline.cfg
    n = int(cfg.n_out)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    res: Dict = {"started": datetime.now().isoformat(timespec="seconds"), "pixel_format_requested": cfg.camera_pixel_format,
                 "standard_checkpoint": str(cfg.tm_checkpoint_path), "standard_dataset": str(cfg.output_h5_path)}

    def save() -> None:
        with open(out_dir / "pbr_bias_check_summary.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(res), f, indent=1)

    rig = ImageCamera(pipeline)
    res["camera_state_before"] = {"image_camera": camera_state(pipeline.cameras[0]), "fourier_camera": camera_state(pipeline.cameras[1])}
    st = res["camera_state_before"]["image_camera"]
    print(f"Image camera {st.get('serial')} ({st.get('model')}): PixelFormat {st.get('PixelFormat')}, BlackLevel {st.get('BlackLevel')}, "
          f"Gain {st.get('Gain')}, GainAuto {st.get('GainAuto')}, ExposureAuto {st.get('ExposureAuto')}")
    e_min = float(rig.cam.ExposureTime.GetMin())
    e0 = max(E0_STANDARD, e_min)
    exposures = [e0 * f for f in FACTORS]
    if e_min <= E_REPOSITORY < e0:
        exposures = [E_REPOSITORY] + exposures
    e_dataset = float(pipeline.exposure_times_vector[0][0])
    res.update({"camera_minimum_exposure": e_min, "e0": e0, "exposures": exposures, "dataset_exposure_image_camera": e_dataset})
    print(f"Minimum exposure {e_min:g}; E0 = {e0:g}; exposures {exposures}")
    save()

    # ---- 1. freshness of the standard TM
    print("\n=== 1. Freshness ===")
    masks_all = pipeline._load_optimized_small_masks()
    idx = held_out_indices(len(masks_all))[:N_FRESH]
    with h5py.File(cfg.output_h5_path, "r") as h5f:
        ds = h5f[cfg.output_group][CROP_NAME]
        if ds.shape[0] != len(masks_all):
            raise RuntimeError(f"The dataset has {ds.shape[0]} frames, the mask file {len(masks_all)}.")
        dataset_frames = np.stack([ds[int(i)] for i in idx]).astype(np.float32)
    dataset_time = datetime.fromtimestamp(Path(cfg.output_h5_path).stat().st_mtime)
    minutes = lambda: (datetime.now() - dataset_time).total_seconds() / 60.0
    res["freshness_mask_indices"] = idx
    fresh = None
    for attempt in (1, 2):
        fresh = freshness(rig, masks_all[idx], dataset_frames, e_dataset)
        np.savez_compressed(out_dir / f"freshness_frames_attempt_{attempt}.npz", frames=fresh.pop("frames"), dataset_frames=dataset_frames, mask_indices=idx)
        fresh["minutes_since_dataset"] = minutes()
        res[f"freshness_attempt_{attempt}"] = fresh
        print(f"  attempt {attempt}: correlation with the dataset frames, 8 masks: mean {fresh['mean']:.3f} (previous mask: {fresh['mean_against_the_previous_mask']:.3f}); "
              f"level now / dataset {fresh['level_now_over_dataset']:.2f}; between different masks {fresh['correlation_between_frames_of_different_masks']:.3f}; "
              f"{minutes():.0f} min after the dataset")
        save()
        if fresh["correlation_between_frames_of_different_masks"] > MAX_BETWEEN_MASKS:
            # On 16 Nov 2026 every mask gave the same speckle (0.999): the SLM did not modulate the light.
            print(f"\nSTOPPED: frames of different SLM masks are correlated at {fresh['correlation_between_frames_of_different_masks']:.3f} "
                  f"(limit {MAX_BETWEEN_MASKS}). The light does not depend on the SLM mask. Nothing else was recorded.")
            res["stopped"] = "the camera frames do not depend on the SLM mask"
            save()
            return res
        if fresh["mean"] >= min_freshness:
            break
    tm_source = "standard checkpoint"
    if fresh["mean"] < min_freshness:
        if not allow_new_dataset:
            print(f"\nSTOPPED: freshness below {min_freshness} twice and --allow-new-dataset was not given. Nothing else was recorded.")
            res["stopped"] = "freshness below the limit twice"
            save()
            return res
        print(f"\n=== Freshness below {min_freshness}: new dataset and new TM with the standard pipeline code, into new files ===")
        tm, info = new_dataset_and_tm(pipeline, rtf, out_dir, stamp)
        res["new_dataset_and_tm"] = info
        tm_source = "new TM of this run"
        with h5py.File(cfg.output_h5_path, "r") as h5f:
            dataset_frames = np.stack([h5f[cfg.output_group][CROP_NAME][int(i)] for i in idx]).astype(np.float32)
        dataset_time = datetime.fromtimestamp(Path(cfg.output_h5_path).stat().st_mtime)
        fresh = freshness(rig, masks_all[idx], dataset_frames, e_dataset)
        fresh.pop("frames")
        fresh["minutes_since_dataset"] = minutes()
        res["freshness_new_dataset"] = fresh
        print(f"  new dataset: {info['acquisition_minutes']:.1f} min; retrieval {info['retrieval_minutes']:.1f} min; freshness {fresh['mean']:.3f}")
        save()
    else:
        tm = torch.load(cfg.tm_checkpoint_path, map_location=device, weights_only=False)
        tm.device = device
    tm.eval()
    for p in tm.parameters():
        p.requires_grad = False
    res["tm_used"] = tm_source

    # ---- 2. targets and patterns (no hardware)
    print("\n=== 2. Five single-focus targets, optimized as in the standard run ===")
    targets = design_targets(pipeline, tm, out_dir)

    # ---- 3. frames
    print("\n=== 3. Frames ===")
    support = pipeline.support_mask("image", (n, n))
    res["support_radius_px"] = float(pipeline.support_radius_px("image"))
    res["targets"] = []
    all_frames = {}
    for t in targets:
        t0 = datetime.now()
        rec = record_target(rig, t["phase_uint8"], exposures)
        np.savez_compressed(out_dir / f"target_{t['index']}_raw_crops_uint16.npz", exposures=np.array(exposures),
                            **{f"exposure_{e:g}": rec["raw_crops"][e] for e in exposures})
        all_frames[t["index"]] = rec["frames"]
        entry = {"index": t["index"], "target_yx": t["target_yx"], "focus_metric_last_iteration": t["focus_metric_last_iteration"],
                 "recorded": t0.isoformat(timespec="seconds"), "minutes_since_dataset": minutes(), "analysis": {}}
        entry["analysis"] = analyse_target(rec["frames"], rec["raw_crops"], support, tuple(t["target_yx"]), e0, rtf.peak_to_background_ratio,
                                           int(cfg.pbr_peak_search_radius_px))
        if E_REPOSITORY in rec["frames"] and E_REPOSITORY != e0:
            entry["analysis_e0_20"] = analyse_target(rec["frames"], rec["raw_crops"], support, tuple(t["target_yx"]), E_REPOSITORY,
                                                     rtf.peak_to_background_ratio, int(cfg.pbr_peak_search_radius_px))
        entry["tm_predicted_pbr"] = float(rtf.peak_to_background_ratio(t["predicted_image"], support, target_yx=tuple(t["target_yx"]),
                                                                       search_radius=int(cfg.pbr_peak_search_radius_px), half_width=1)[0])
        res["targets"].append(entry)
        a = entry["analysis"]
        print(f"  target {t['index']} at {t['target_yx']}: peak {a['peak_e0_counts_mean']:.0f} counts at E0, background {a['background_e0_mean_counts']:.2f} counts "
              f"({a['background_e0_zero_fraction']:.2f} zeros); PBR (i) {a['pbr_i']:.0f}, (ii) {a.get('pbr_ii', float('nan')):.0f}, (iii) {a.get('pbr_iii', float('nan')):.0f}")
        save()
    np.savez_compressed(out_dir / "focus_frames_128.npz", exposures=np.array(exposures), targets_yx=np.array([t["target_yx"] for t in targets]),
                        **{f"target_{k}_exposure_{e:g}": all_frames[k][e] for k in all_frames for e in exposures})
    res["summary"] = summarise([t["analysis"] for t in res["targets"]])
    if all("analysis_e0_20" in t for t in res["targets"]):
        res["summary_e0_20"] = summarise([t["analysis_e0_20"] for t in res["targets"]])
    res["warm_up_frames_discarded"] = rig.n_warmup

    # ---- freshness at the end (information only) and the camera state during the frames
    end = freshness(rig, masks_all[idx], dataset_frames, e_dataset)
    end.pop("frames")
    end["minutes_since_dataset"] = minutes()
    res["freshness_end"] = end
    res["camera_state_after"] = {"image_camera": camera_state(pipeline.cameras[0])}
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    print_report(res)
    print(f"freshness at the end: {end['mean']:.3f}; warm-up frames discarded: {rig.n_warmup}")
    print(f"\nSaved to {out_dir}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-new-dataset", action="store_true",
                    help="if the freshness is below the limit twice, record a new dataset and retrieve a new TM into new files")
    ap.add_argument("--min-freshness", type=float, default=0.9)
    args = ap.parse_args()

    import run_tm_focus_optimization as rtf                               # hardware libraries are loaded here

    pipeline = rtf.TMFocusPipeline(rtf.PipelineConfig())
    out_dir = Path(pipeline.cfg.output_dir) / f"pbr_bias_check_{datetime.now().strftime('%Y%m%d_%H%M')}"
    pipeline.setup()                                                      # SLM, cameras, Mono12 (as the standard run)
    try:
        run_check(pipeline, rtf, out_dir, args.allow_new_dataset, args.min_freshness)
    finally:
        pipeline.teardown()                                               # cameras back to Mono8 at the minimum exposure, SLM released


if __name__ == "__main__":
    main()
