"""
Stage A of the plan of 12 Nov 2026 (offline, no hardware): steps A1, A2 and A3.

    python stageA_offline_analysis.py                              # current TM checkpoint, A1 + A2 + A3
    python stageA_offline_analysis.py --checkpoint <file> --tag b0.4 --only-a3

Reads (never writes) the 12:34 dataset, the TM checkpoint and the saved run folders
tm_focus_outputs/polarization_axes (image-plane run, 16:01) and
tm_focus_outputs/polarization_axes_fourier (Fourier-plane run, 16:22).
Writes tm_focus_outputs/stageA_offline/stageA_<tag>.json.

Conventions
* Stage angles as recorded; lab x = stage 0 deg, lab y = stage 90 deg.
* Correlation = Pearson correlation of intensities over the stated pixels.
* tau = P_0 / (P_0 + P_90): fraction of a focus that passes the polarizer at stage 0.
  Separation = tau_A - tau_B. "Before the arm": powers divided by the arm weights (1 + d, 1 - d).
"""
from __future__ import annotations

import argparse
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

import polarizer_arm_estimator as pae          # noqa: E402
import run_polarization_axes as rpa            # noqa: E402  (imports no hardware library at module level)

OUT_FOURIER = TM_LIB / "tm_focus_outputs" / "polarization_axes_fourier"
OUT_IMAGE = TM_LIB / "tm_focus_outputs" / "polarization_axes"
H5_1234 = TM_LIB / "measurement_focus.h5"
MASKS_H5 = TM_LIB / "tm_motor_pcc_outputs" / "spectral_init_slm_masks.h5"
PROBE_IDX = [1627, 3082, 3840, 5128]           # probe masks used by both runs of 12 Nov
N = 128
SUPPORT_RADIUS = 58.1                          # px: fiber core (image plane) and NA disk (Fourier plane)
FIT_RADIUS = 50.1                              # px: fit region of run_polarization_axes.solve_frame in the Fourier run
D_FOCI_DEFAULT = -0.673                        # mean d of the two saved patterns (step A2), used when A2 is skipped


def disk(radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:N, :N]
    return (yy - (N - 1) / 2.0) ** 2 + (xx - (N - 1) / 2.0) ** 2 <= radius ** 2


def rng_txt(values, fmt="{:.3f}") -> str:
    v = np.asarray(values, dtype=float).ravel()
    return f"{fmt.format(v.min())} to {fmt.format(v.max())} (mean {fmt.format(v.mean())})"


def speckle_contrast(frames: np.ndarray, support: np.ndarray, saturation: float = 4095.0) -> dict:
    """frames: (K, H, W). Contrast = std / mean over the support pixels of one frame, averaged over frames."""
    f = frames.astype(np.float64)
    env = f.mean(axis=0)                                           # mean image over all probe masks
    inside = f[:, support]
    raw = inside.std(axis=1) / inside.mean(axis=1)
    flat = inside / env[support][None, :]                          # divide out the mean envelope
    flat_c = flat.std(axis=1) / flat.mean(axis=1)
    per_pixel = (f.std(axis=0) / np.maximum(env, 1e-9))[support]   # contrast over the masks, pixel by pixel
    return {"contrast_envelope_removed": float(flat_c.mean()), "contrast_raw": float(raw.mean()),
            "contrast_over_masks_per_pixel": float(per_pixel.mean()),
            "mean_level_in_support": float(inside.mean()),
            "saturated_pixel_fraction": float((inside >= saturation).mean())}


def a1(result: dict) -> None:
    print("\n=== A1. Effect of the polarizer on the image ===")
    summ = json.load(open(OUT_FOURIER / "polarization_axes_summary.json"))
    d = np.load(OUT_FOURIER / "polarizer_speckle_frames.npz")
    raw, angles, masks = d["frames"].astype(np.float64), [float(a) for a in d["angles_deg"]], d["slm_masks_uint8"]
    shifts = np.array(summ["frame"]["shifts_px"])
    reg = rpa.register_stack(raw, shifts)                                           # (4 angles, 4 masks, N, N)
    with h5py.File(MASKS_H5, "r") as h5:
        probe = np.asarray(h5["mode_excitation_patterns"]["phase_masks_small_uint8"][PROBE_IDX], dtype=np.uint8)
    assert np.array_equal(probe, masks), "saved SLM masks are not the probe masks " + str(PROBE_IDX)
    with h5py.File(H5_1234, "r") as h5:
        g = h5["experiment_0"]
        four_then = np.asarray(g["transmission_fourier"][PROBE_IDX], dtype=np.float64)                 # no flip
        im_then = np.flip(np.asarray(g["transmission_image"][PROBE_IDX], dtype=np.float64), axis=(-2, -1))
        four_all = np.asarray(g["transmission_fourier"][:], dtype=np.float32)
        im_all = np.asarray(g["transmission_image"][:], dtype=np.float32)
    fit, support = disk(FIT_RADIUS), disk(SUPPORT_RADIUS)
    i0, i45, i90, i135 = (angles.index(a) for a in (0.0, 45.0, 90.0, 135.0))
    sum_a, sum_b = reg[i0] + reg[i90], reg[i45] + reg[i135]

    pcc_a = [rpa.pcc(sum_a[k][fit], four_then[k][fit]) for k in range(4)]
    pcc_b = [rpa.pcc(sum_b[k][fit], four_then[k][fit]) for k in range(4)]
    pcc_ab = [rpa.pcc(sum_a[k][fit], sum_b[k][fit]) for k in range(4)]
    single = [[rpa.pcc(reg[ia][k][fit], four_then[k][fit]) for k in range(4)] for ia in range(4)]
    level_a = [float(sum_a[k][fit].mean() / four_then[k][fit].mean()) for k in range(4)]
    level_b = [float(sum_b[k][fit].mean() / four_then[k][fit].mean()) for k in range(4)]
    # displacement of each polarizer angle against the 12:34 frames of the same masks (no model involved)
    shifts_free = np.array([rpa.estimate_shift(raw[ia], four_then, 12.0)[:2] for ia in range(4)])
    # residual displacement of the registered sum against the 12:34 frames
    resid = rpa.estimate_shift(sum_a, four_then, 4.0)[:2]

    di = np.load(OUT_IMAGE / "polarizer_speckle_frames.npz")
    assert np.array_equal(di["slm_masks_uint8"], masks)
    im_now = di["frames"].astype(np.float64)                                        # (4 angles, 4 masks), polarizer not in this arm
    im_pcc = [[rpa.pcc(im_now[ia][k][support], im_then[k][support]) for k in range(4)] for ia in range(im_now.shape[0])]
    im_pcc_mean_frame = [rpa.pcc(im_now[:, k].mean(axis=0)[support], im_then[k][support]) for k in range(4)]
    im_repeat = [rpa.pcc(im_now[0][k][support], im_now[ia][k][support]) for ia in range(1, im_now.shape[0]) for k in range(4)]
    im_level = float(np.mean([im_now[:, k][:, support].mean() / im_then[k][support].mean() for k in range(4)]))

    c_four, c_im = speckle_contrast(four_all, support), speckle_contrast(im_all, support)
    # Reference for the Fourier camera: contrast of the same four masks behind the polarizer (one polarization),
    # with the mean envelope of the 12:34 frames divided out, inside the fit region.
    env = four_all.mean(axis=0).astype(np.float64)

    def contrast4(frames4):
        x = np.stack([f[fit] / env[fit] for f in frames4])
        return float((x.std(axis=1) / x.mean(axis=1)).mean())

    c4 = {"no_polarizer_1234": contrast4(four_then), "sum_0_90": contrast4(sum_a), "sum_45_135": contrast4(sum_b),
          "single_angles_0_45_90_135": [contrast4(reg[ia]) for ia in (i0, i45, i90, i135)]}
    mean_light = [float(reg[ia][:, fit].mean()) for ia in (i0, i45, i90, i135)]
    s1m, s2m = mean_light[0] - mean_light[2], mean_light[1] - mean_light[3]
    s0m = 0.5 * sum(mean_light)
    dolp_mean = float(np.hypot(s1m, s2m) / s0m)
    az_mean = float(np.degrees(0.5 * np.arctan2(s2m, s1m)) % 180.0)

    print(f"Fourier plane, I_0 + I_90 (registered, 16:22) against the 12:34 frame, same mask, r <= {FIT_RADIUS:.0f} px: "
          f"{rng_txt(pcc_a)}; per mask {[round(v, 3) for v in pcc_a]}")
    print(f"Fourier plane, I_45 + I_135 against the 12:34 frame: {rng_txt(pcc_b)}")
    print(f"Fourier plane, I_0 + I_90 against I_45 + I_135 (same run): {rng_txt(pcc_ab)}")
    print(f"Fourier plane, single angles against the 12:34 frame (0/45/90/135 deg, mean over masks): "
          f"{[round(float(np.mean(s)), 3) for s in single]}")
    print(f"Image plane, 16:01 frame against the 12:34 frame, same mask, core (r <= {SUPPORT_RADIUS} px): {rng_txt(im_pcc)}; "
          f"mean of the 4 repeats per mask {[round(v, 3) for v in im_pcc_mean_frame]}")
    print(f"Image plane, repeat-to-repeat within the 16:01 run: {rng_txt(im_repeat)}; level now / then {im_level:.3f}")
    print(f"Level (I_0 + I_90) / (12:34 frame), same exposure 675: {rng_txt(level_a)}; (I_45 + I_135) / (12:34 frame): {rng_txt(level_b)}")
    print("Displacement (dy, dx) px per stage angle, from the model (run) | from the 12:34 frames (no model) | difference:")
    for ia, a in enumerate(angles):
        diff = float(np.hypot(*(shifts[ia] - shifts_free[ia])))
        print(f"  {a:5.0f} deg: ({shifts[ia][0]:+6.2f}, {shifts[ia][1]:+6.2f}) | ({shifts_free[ia][0]:+6.2f}, {shifts_free[ia][1]:+6.2f}) | {diff:.2f} px")
    print(f"Residual displacement of the registered sum against the 12:34 frames: ({resid[0]:+.2f}, {resid[1]:+.2f}) px")
    for name, c in (("Fourier", c_four), ("image", c_im)):
        print(f"Speckle contrast, 12:34 {name} frames (6032 masks, support r <= {SUPPORT_RADIUS} px): "
              f"{c['contrast_envelope_removed']:.3f} with the mean envelope divided out, {c['contrast_raw']:.3f} raw, "
              f"{c['contrast_over_masks_per_pixel']:.3f} over the masks per pixel; mean level {c['mean_level_in_support']:.1f} counts, "
              f"saturated pixels {100 * c['saturated_pixel_fraction']:.3f} %")
    print("  (two independent polarization components with mean powers in the ratio (1+P):(1-P) give sqrt((1+P^2)/2): "
          "0.707 for P = 0, 0.78 for P = 0.47, 0.85 for P = 0.67, 1 for P = 1)")
    print(f"Same four masks on the Fourier camera, r <= {FIT_RADIUS:.0f} px, envelope divided out: no polarizer (12:34) {c4['no_polarizer_1234']:.3f}; "
          f"I_0 + I_90 {c4['sum_0_90']:.3f}; I_45 + I_135 {c4['sum_45_135']:.3f}; single angles 0/45/90/135 deg "
          f"{[round(v, 3) for v in c4['single_angles_0_45_90_135']]}")
    print(f"Mean light behind the polarizer, r <= {FIT_RADIUS:.0f} px (0/45/90/135 deg): {[round(v, 1) for v in mean_light]} counts; "
          f"degree of linear polarization {dolp_mean:.3f}, azimuth {az_mean:.1f} deg; share along stage 90: "
          f"{mean_light[2] / (mean_light[0] + mean_light[2]):.3f}")
    result["A1"] = {
        "fourier_sum_0_90_vs_1234_pcc": pcc_a, "fourier_sum_45_135_vs_1234_pcc": pcc_b, "fourier_sum_0_90_vs_sum_45_135_pcc": pcc_ab,
        "fourier_single_angle_vs_1234_pcc": single, "level_sum_0_90_over_1234": level_a, "level_sum_45_135_over_1234": level_b,
        "image_then_vs_now_pcc": im_pcc, "image_then_vs_now_pcc_mean_of_repeats": im_pcc_mean_frame,
        "image_repeat_pcc": im_repeat, "image_level_now_over_then": im_level,
        "shifts_from_model_px": shifts.tolist(), "shifts_from_1234_frames_px": shifts_free.tolist(),
        "residual_shift_of_sum_px": [float(resid[0]), float(resid[1])],
        "speckle_contrast_fourier": c_four, "speckle_contrast_image": c_im, "contrast_four_masks_fourier_camera": c4,
        "mean_light_behind_polarizer_counts": mean_light, "mean_light_dolp": dolp_mean, "mean_light_azimuth_deg": az_mean,
        "regions": {"fourier_px_radius": FIT_RADIUS, "image_px_radius": SUPPORT_RADIUS},
    }


def a2(result: dict) -> None:
    print("\n=== A2. Diattenuation of the polarizer arm from the saved foci ===")
    summ = json.load(open(OUT_FOURIER / "polarization_axes_summary.json"))["two_foci"]
    rows = {}
    for tag in ("tm_axes_uncorrected", "lab_axes"):
        m = summ[tag]["measured"]
        (pax, pay), (pbx, pby) = m["A"]["window_power"], m["B"]["window_power"]
        p_a, p_b = (pax - pay) / (pax + pay), (pbx - pby) / (pbx + pby)
        rows[tag] = {"powers_A_x_y": [pax, pay], "powers_B_x_y": [pbx, pby], "p_A": p_a, "p_B": p_b,
                     "d": pae.diattenuation_from_orthogonal_pair(p_a, p_b), "dark_level": m["dark_level"]}
    d_mean = float(np.mean([rows[t]["d"] for t in rows]))
    for tag, r in rows.items():
        (pax, pay), (pbx, pby) = r["powers_A_x_y"], r["powers_B_x_y"]
        r["raw"] = {"tau_A": pax / (pax + pay), "tau_B": pbx / (pbx + pby)}
        r["raw"]["separation"] = r["raw"]["tau_A"] - r["raw"]["tau_B"]
        for label, dd in (("before_arm_d_mean", d_mean), ("before_arm_d_own", r["d"])):
            est = {"weights_xy": [1.0 + dd, 1.0 - dd]}
            ta, tb = pae.pass_fraction_before_arm(pax, pay, est), pae.pass_fraction_before_arm(pbx, pby, est)
            r[label] = {"d_used": dd, "tau_A": ta, "tau_B": tb, "separation": ta - tb}
        print(f"{tag}: d = {r['d']:+.3f} (power ratio stage 90 / stage 0 = {(1 - r['d']) / (1 + r['d']):.2f})")
        print(f"  raw:            A passes {r['raw']['tau_A']:.3f} at stage 0, B passes {r['raw']['tau_B']:.3f}; separation {r['raw']['separation']:+.3f}")
        print(f"  before the arm: A {r['before_arm_d_mean']['tau_A']:.3f}, B {r['before_arm_d_mean']['tau_B']:.3f}; separation "
              f"{r['before_arm_d_mean']['separation']:+.3f} (d = {d_mean:+.3f}, mean of both patterns); "
              f"with the pattern's own d: {r['before_arm_d_own']['separation']:+.3f}")
    print(f"Mean d of the two patterns: {d_mean:+.3f}. Window powers are 3x3 sums; the dark level of these frames is 0 counts.")
    result["A2"] = {"patterns": rows, "d_mean": d_mean}


def load_tm(path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tm = torch.load(path, map_location=device, weights_only=False)
    tm.device = device
    tm.eval()
    for p in tm.parameters():
        p.requires_grad = False
    return tm


def a3(result: dict, checkpoint: Path) -> None:
    print(f"\n=== A3. Speckle estimate with diattenuation ({checkpoint.name}) ===")
    summ = json.load(open(OUT_FOURIER / "polarization_axes_summary.json"))
    d = np.load(OUT_FOURIER / "polarizer_speckle_frames.npz")
    raw, angles, masks = d["frames"].astype(np.float64), [float(a) for a in d["angles_deg"]], d["slm_masks_uint8"]
    tm = load_tm(checkpoint)
    weights = getattr(tm, "fourier_pol_weights", (1.0, 1.0))
    with torch.no_grad():
        e_model = rpa.model_plane_field(tm, rpa._phase_from_uint8(tm, masks), "fourier").cpu().numpy()   # (K, 2, N, N)
    fit = disk(FIT_RADIUS)
    shifts = np.array(summ["frame"]["shifts_px"])
    s0_cam = weights[0] * np.abs(e_model[:, 0]) ** 2 + weights[1] * np.abs(e_model[:, 1]) ** 2
    if tuple(weights) != (1.0, 1.0):
        # a different TM: register again, against its own Fourier-camera prediction
        shifts = np.array([rpa.estimate_shift(raw[ia], s0_cam, 12.0)[:2] for ia in range(len(angles))])
    reg = rpa.register_stack(raw, shifts)
    est = pae.estimate_frame_with_diattenuation(reg, angles, e_model, fit)
    if tuple(weights) != (1.0, 1.0):
        for _ in range(2):                                         # refine the registration against the predicted polarizer images
            pred = np.stack([np.stack([pae.predict_polarizer_image(e_model[k], est, a) for k in range(len(masks))]) for a in angles])
            shifts = np.array([rpa.estimate_shift(raw[ia], pred[ia], 12.0)[:2] for ia in range(len(angles))])
            reg = rpa.register_stack(raw, shifts)
            est = pae.estimate_frame_with_diattenuation(reg, angles, e_model, fit)
    est_axis0 = pae.estimate_frame_with_diattenuation(reg, angles, e_model, fit, fit_axis=False)

    def pcc_per_angle(e):
        return [float(np.mean([rpa.pcc(pae.predict_polarizer_image(e_model[k], e, a)[fit], reg[ia, k][fit])
                               for k in range(len(masks))])) for ia, a in enumerate(angles)]

    pcc_new = pcc_per_angle(est)
    u_old = np.load(OUT_FOURIER / "jones_matrix_U.npy")
    old = {"U": u_old, "d": 0.0, "theta0_deg": 0.0}
    pcc_old = pcc_per_angle(old)
    old_explained = float(rpa.estimate_frame(reg, angles, e_model, fit)["explained_fraction"])
    r = est["R"]
    r_old = rpa.estimate_frame(reg, angles, e_model, fit)["R"]
    ang_r = float(np.degrees(np.arccos(np.clip((np.trace(r @ r_old.T) - 1) / 2, -1, 1))))
    ang_r1 = float(np.degrees(np.arccos(np.clip(r[0] @ r_old[0], -1, 1))))
    print(f"TM: Fourier-camera weights of the model {tuple(round(float(w), 3) for w in weights)}")
    print(f"d = {est['d']:+.3f} (S0 term of I_0 - I_90); |d| = {est['d_from_s2_amplitude']:.3f} (amplitude of I_45 - I_135); "
          f"d = {est['d_from_total_intensity']:+.3f} (polarization dependence of I_0 + I_90)")
    print(f"theta0 = {est['theta0_deg']:+.2f} deg (stage angle of the arm axis nearest to stage 0); S0 leak into I_45 - I_135: {est['s0_leak_into_a2']:+.3f}")
    print(f"with theta0 fixed at 0: d = {est_axis0['d']:+.3f}, explained fraction {est_axis0['explained_fraction']:.3f}")
    print(f"explained fraction: {est['explained_fraction']:.3f} (unitary-only model on the same data: {old_explained:.3f}; run of 16:22: "
          f"{summ['frame']['explained_fraction']:.3f})")
    print(f"correlation per angle (0/45/90/135 deg): {[round(v, 3) for v in pcc_new]} "
          f"(unitary U of the run: {[round(v, 3) for v in pcc_old]})")
    print(f"x' of the TM at the polarizer, before the arm: azimuth {est['xprime_azimuth_deg']:.1f} deg, ellipticity {est['xprime_ellipticity_deg']:+.1f} deg")
    print("R (rows: lab S1, S2, S3; columns: model S1, S2, S3):")
    for row in r:
        print("   [" + ", ".join(f"{v:+.3f}" for v in row) + "]")
    print(f"|U| = [[{abs(est['U'][0, 0]):.3f}, {abs(est['U'][0, 1]):.3f}], [{abs(est['U'][1, 0]):.3f}, {abs(est['U'][1, 1]):.3f}]]"
          f" (diagonal U: [[1, 0], [0, 1]]; swapped axes: [[0, 1], [1, 0]])")
    print(f"angle between this R and the unitary-only R on the Poincare sphere: {ang_r:.1f} deg; between their first rows: {ang_r1:.1f} deg")
    if tuple(weights) != (1.0, 1.0):
        print("displacements (dy, dx) px re-estimated for this TM: " + ", ".join(f"({s[0]:+.2f}, {s[1]:+.2f})" for s in shifts))

    # --- the four saved foci ---
    d_foci = result["A2"]["d_mean"] if "A2" in result else D_FOCI_DEFAULT
    est_dfoci = {"U": est["U"], "d": d_foci, "theta0_deg": est["theta0_deg"]}
    foci = {}
    yy, xx = np.mgrid[:N, :N]
    print(f"Saved foci, fraction that passes at stage 0 (tau): measured | this model | this model's R with d = {d_foci:+.3f} from the foci (A2) | unitary U of the run")
    for tag in ("tm_axes_uncorrected", "lab_axes"):
        m = summ["two_foci"][tag]["measured"]
        u8 = np.load(OUT_FOURIER / f"two_foci_{tag}_phase_uint8.npy")
        with torch.no_grad():
            ef = rpa.model_plane_field(tm, rpa._phase_from_uint8(tm, u8[None]), "fourier")[0].cpu().numpy()
        foci[tag] = {}
        for name, tgt_key in (("A", "target_A_x_pol_yx"), ("B", "target_B_y_pol_yx")):
            tgt = summ["two_foci"][tgt_key]
            near = (np.abs(yy - tgt[0]) <= 3) & (np.abs(xx - tgt[1]) <= 3)
            s0 = (np.abs(ef) ** 2).sum(axis=0)
            py, px = np.unravel_index(int(np.argmax(np.where(near, s0, -np.inf))), s0.shape)
            win = (np.abs(yy - py) <= 1) & (np.abs(xx - px) <= 1)
            p0, p90 = (float(pae.predict_polarizer_image(ef, est, a)[win].sum()) for a in (0.0, 90.0))
            q0, q90 = (float(pae.predict_polarizer_image(ef, old, a)[win].sum()) for a in (0.0, 90.0))
            f0, f90 = (float(pae.predict_polarizer_image(ef, est_dfoci, a)[win].sum()) for a in (0.0, 90.0))
            before = pae.field_at_polarizer(ef, {"U": est["U"], "d": 0.0})
            b0, b90 = float((np.abs(before[0]) ** 2)[win].sum()), float((np.abs(before[1]) ** 2)[win].sum())
            mp0, mp90 = m[name]["window_power"]
            foci[tag][name] = {"measured_tau": mp0 / (mp0 + mp90), "predicted_tau": p0 / (p0 + p90), "predicted_tau_unitary_run": q0 / (q0 + q90),
                               "predicted_tau_R_speckle_d_foci": f0 / (f0 + f90),
                               "predicted_tau_before_arm_in_estimator_axes": b0 / (b0 + b90), "model_peak_yx": [int(py), int(px)],
                               "model_window_share_of_total": float(s0[win].sum() / s0.sum())}
            f = foci[tag][name]
            print(f"  {tag:20s} focus {name}: {f['measured_tau']:.3f} | {f['predicted_tau']:.3f} | {f['predicted_tau_R_speckle_d_foci']:.3f} | {f['predicted_tau_unitary_run']:.3f}")
        sep_m = foci[tag]["A"]["measured_tau"] - foci[tag]["B"]["measured_tau"]
        sep_p = foci[tag]["A"]["predicted_tau"] - foci[tag]["B"]["predicted_tau"]
        sep_f = foci[tag]["A"]["predicted_tau_R_speckle_d_foci"] - foci[tag]["B"]["predicted_tau_R_speckle_d_foci"]
        foci[tag]["separation_measured"], foci[tag]["separation_predicted"], foci[tag]["separation_predicted_R_speckle_d_foci"] = sep_m, sep_p, sep_f
        print(f"  {tag:20s} separation: measured {sep_m:+.3f} | this model {sep_p:+.3f} | with d from the foci {sep_f:+.3f}")
    tags = ("tm_axes_uncorrected", "lab_axes")
    err = [abs(foci[t][n]["predicted_tau"] - foci[t][n]["measured_tau"]) for t in tags for n in ("A", "B")]
    err_old = [abs(foci[t][n]["predicted_tau_unitary_run"] - foci[t][n]["measured_tau"]) for t in tags for n in ("A", "B")]
    err_f = [abs(foci[t][n]["predicted_tau_R_speckle_d_foci"] - foci[t][n]["measured_tau"]) for t in tags for n in ("A", "B")]
    print(f"  mean |predicted - measured| over the four foci: {np.mean(err):.3f} this model; {np.mean(err_f):.3f} with d from the foci; "
          f"{np.mean(err_old):.3f} unitary U of the run")
    result["A3"] = {
        "checkpoint": str(checkpoint), "fourier_pol_weights_of_model": [float(w) for w in weights],
        "d": est["d"], "d_from_s2_amplitude": est["d_from_s2_amplitude"], "d_from_total_intensity": est["d_from_total_intensity"],
        "theta0_deg": est["theta0_deg"], "s0_leak_into_a2": est["s0_leak_into_a2"], "explained_fraction": est["explained_fraction"],
        "explained_fraction_unitary_only_same_data": old_explained, "pcc_per_angle": pcc_new, "pcc_per_angle_unitary_run": pcc_old,
        "d_theta0_fixed_0": est_axis0["d"], "explained_fraction_theta0_fixed_0": est_axis0["explained_fraction"],
        "R": r.tolist(), "U_abs": np.abs(est["U"]).tolist(), "U_real": est["U"].real.tolist(), "U_imag": est["U"].imag.tolist(),
        "xprime_azimuth_deg": est["xprime_azimuth_deg"], "xprime_ellipticity_deg": est["xprime_ellipticity_deg"],
        "angle_to_unitary_only_R_deg": ang_r, "angle_between_first_rows_deg": ang_r1, "shifts_px": shifts.tolist(),
        "foci": foci, "foci_mean_abs_error": float(np.mean(err)), "foci_mean_abs_error_unitary_run": float(np.mean(err_old)),
        "foci_mean_abs_error_R_speckle_d_foci": float(np.mean(err_f)), "d_foci_used": d_foci,
        "fit_region_radius_px": FIT_RADIUS,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(TM_LIB / "tm_focus_outputs" / "tm_system_checkpoint.pt"))
    ap.add_argument("--tag", default="current_tm")
    ap.add_argument("--only-a3", action="store_true")
    args = ap.parse_args()
    out_dir = TM_LIB / "tm_focus_outputs" / "stageA_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    if not args.only_a3:
        a1(result)
        a2(result)
    a3(result, Path(args.checkpoint))
    path = out_dir / f"stageA_{args.tag}.json"
    json.dump(result, open(path, "w"), indent=1)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
