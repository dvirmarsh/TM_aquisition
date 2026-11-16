"""
Phase between the two polarization components of a TM retrieved with the camera weight b.

With the weight in the forward model, the retrieved axes x', y' coincide with the axes of the beam
splitter, and the field in front of a polarizer at the Fourier-plane camera is

    E_pol = c * ( sqrt(1+b) * exp(i*delta) * E_x ,  sqrt(1-b) * E_y ),

where (E_x, E_y) is the Fourier-plane field predicted by the retrieved TM and delta is the one
phase that the retrieval leaves free. A polarizer at angle a from x transmits

    I_a = g * [ cos^2(a) * A + sin^2(a) * B + sin(2a) * ( C*cos(delta) - S*sin(delta) ) ],
    A = (1+b)|E_x|^2,  B = (1-b)|E_y|^2,  C + i*S = sqrt(1-b^2) * E_x * conj(E_y).

One image at a = 45 deg of one speckle pattern is therefore linear in (cos delta, sin delta):
a least-squares fit over the pixels gives delta = atan2(c_S, c_C). Images of several SLM patterns
can be pooled in the same fit, and a second angle (135 deg) removes the unpolarized part.

This script retrieves the TM of a scaled-down fiber (repo forward model and fast ADMM), simulates
polarizer images of single SLM patterns, and compares the estimate with the true phase.

Numbers of Supplementary Section S7 (about 3 minutes each on two CPU cores):
    python paper/simulations/sim_xy_phase.py --seeds 1,2,3,4,5,6,7,8,9,10 --out paper/simulations/results_xy_phase_ideal.json
    python paper/simulations/sim_xy_phase.py --seeds 1,2,3,4,5,6,7,8,9,10 --mu 0.97 --data-noise 0.1 --out paper/simulations/results_xy_phase_drift.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from TM_retrieve import TMMeasurementSystem                              # noqa: E402
from TM_retrieval_algos import optimize_tm_admm_dual_plane               # noqa: E402

torch.set_num_threads(2)


def build_system(core_radius: float = 8e-6, n_grid: int = 48, seed: int = 0, na: float = 0.22, cache_dir=None):
    """Scaled-down copy of the experimental optics (core radius 8 um, 156 guided modes)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    wavelength, core_index, magnification = 632.8e-9, 1.457, 50.0
    obj_f, dx_cam = 200e-3 / magnification, 3.45e-6
    dx_im, dx_four, dx_slm = dx_cam / magnification, dx_cam, 8e-6 * (150.0 / 400.0)
    margin, half = 1.1, n_grid / 2.0
    n_pix_im, n_pix_four = round(margin * core_radius / dx_im), round(margin * na * obj_f / dx_four)
    dx_final = {"im": dx_im * n_pix_im / half, "four": dx_four * n_pix_four / half, "slm": dx_slm * 320 / half}
    mult_factor = {"tran": dx_final["im"] ** 2 / (wavelength * obj_f), "slm": dx_final["slm"] ** 2 / (wavelength * obj_f)}
    fiber_params = {"a": core_radius, "n1": core_index, "n0": float(np.sqrt(core_index ** 2 - na ** 2)), "wavelength": wavelength}
    rng = half * dx_final["im"]
    grid_params = {"x_range": (-rng, rng), "y_range": (-rng, rng), "nx": n_grid, "ny": n_grid}
    a_param = {"slm": 2 * np.pi * dx_final["slm"] * dx_final["im"] / (wavelength * obj_f),
               "tran": 2 * np.pi * dx_final["im"] * dx_final["four"] / (wavelength * obj_f)}
    c0 = (n_grid - 1) / 2
    zoomfft_params = {side: {"a_init": a_param[side], "n0_init": c0, "m0_init": c0, "k0_init": c0, "l0_init": c0}
                      for side in ("slm", "tran")}
    cache = Path(cache_dir) if cache_dir else REPO_ROOT / "data" / "mode_fields"
    cache.mkdir(parents=True, exist_ok=True)
    return TMMeasurementSystem(fiber_params, grid_params, device="cpu", mult_factor=mult_factor,
                               zoomfft_params=zoomfft_params, zernike_n_modes=8, dx=dx_final,
                               lab_data=False, cache_dir=str(cache))


def set_model_weights(s, b: float) -> None:
    """Camera weights of the paper in the cached operators: image (1, 1), Fourier (1 + b, 1 - b)."""
    s.calibrate()
    with torch.no_grad():
        for name in ("pre", "post"):
            getattr(s, f"_cal_fourier_{name}_x").mul_(float(np.sqrt(1.0 + b)))
            getattr(s, f"_cal_fourier_{name}_y").mul_(float(np.sqrt(1.0 - b)))


def polarizer_image(e_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    """Intensity behind an ideal linear polarizer at angle_deg from x; e_xy: (2, H, W)."""
    a = np.radians(angle_deg)
    return np.abs(np.cos(a) * e_xy[0] + np.sin(a) * e_xy[1]) ** 2


def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def estimate_delta_one_angle(imgs, e_cam, mask, angle_deg=45.0, free_axis_terms=True):
    """
    delta from polarizer images at ONE angle. imgs: (K, H, W), one image per SLM pattern (K = 1 is a
    single image); e_cam: (K, 2, H, W) model field at the Fourier camera with the weights included,
    i.e. (sqrt(1+b) E_x, sqrt(1-b) E_y). All pixels of all K images enter one least-squares fit.
    free_axis_terms=True fits A and B with separate coefficients, which absorbs a small error of the
    polarizer angle; False uses the nominal cos^2(a) A + sin^2(a) B.
    """
    a = np.radians(angle_deg)
    aa, bb = (np.abs(e_cam[:, 0]) ** 2)[:, mask].ravel(), (np.abs(e_cam[:, 1]) ** 2)[:, mask].ravel()
    p = (e_cam[:, 0] * e_cam[:, 1].conj())[:, mask].ravel()
    y = imgs[:, mask].ravel()
    one = np.ones_like(y)
    if free_axis_terms:
        x = np.stack([one, aa, bb, np.sin(2 * a) * p.real, -np.sin(2 * a) * p.imag], axis=1)
    else:
        x = np.stack([one, np.cos(a) ** 2 * aa + np.sin(a) ** 2 * bb, np.sin(2 * a) * p.real, -np.sin(2 * a) * p.imag], axis=1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(np.degrees(np.arctan2(coef[-1], coef[-2])))


def estimate_delta_two_angles(imgs45, imgs135, e_cam, mask):
    """delta from I_45 - I_135 = 2 g (C cos(delta) - S sin(delta)); the unpolarized part and any offset cancel."""
    p = (e_cam[:, 0] * e_cam[:, 1].conj())[:, mask].ravel()
    d = (imgs45 - imgs135)[:, mask].ravel()
    coef, *_ = np.linalg.lstsq(np.stack([p.real, -p.imag], axis=1), d, rcond=None)
    return float(np.degrees(np.arctan2(coef[1], coef[0])))


def pcc(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def run(seed, b_true, n_grid, budget, data_noise, mu, noises, n_patterns):
    out = {"seed": seed, "b": b_true, "mu": mu}
    s = build_system(n_grid=n_grid, seed=seed)
    m, n = len(s.guided_modes), n_grid
    out["modes"] = m
    t0 = s.real_tm_tensor.clone()
    rng = np.random.default_rng(1000 + seed)

    def field(phase, tm):
        with torch.no_grad():
            return s.forward_model_slm(phase, tm=tm, use_internal_params=False, apply_slm_amplitude=False,
                                       return_complex_field=True)

    def cameras(e):
        i_im = e[:, 0, 0].abs() ** 2 + e[:, 0, 1].abs() ** 2
        i_f = (1 + b_true) * e[:, 1, 0].abs() ** 2 + (1 - b_true) * e[:, 1, 1].abs() ** 2
        return torch.stack([i_im, i_f], dim=1)

    torch.manual_seed(seed + 1)
    slm = torch.rand(int(round(budget * m)), n, n) * 2 * np.pi
    intens = cameras(field(slm, t0))
    if data_noise:
        intens = (intens * (1.0 + data_noise * torch.randn_like(intens))).clamp(min=0)

    # retrieval with the camera weight in the forward model
    set_model_weights(s, b_true)
    hist = optimize_tm_admm_dual_plane(s, s.slm_to_input_modes(slm), intens, num_epochs=90, planes="both",
                                       init_method="default_beta_prior")
    out["final_loss"] = float(hist["epoch_losses"][-1])

    # fiber at the time of the polarizer image (field correlation mu with the acquisition)
    gg = torch.Generator().manual_seed(4242 + seed)
    noise_tm = torch.complex(torch.randn(t0.shape, generator=gg), torch.randn(t0.shape, generator=gg)) / np.sqrt(2.0)
    t_now = mu * t0 + float(np.sqrt(max(1.0 - mu ** 2, 0.0))) * noise_tm * t0.abs().pow(2).mean().sqrt()

    g = torch.Generator().manual_seed(999 + seed)
    test = torch.rand(n_patterns, n, n, generator=g) * 2 * np.pi
    with torch.no_grad():
        e_cam = s.forward_model_slm(test, return_complex_field=True)[:, 1].numpy()      # weights included
    # true field in front of the polarizer; phi stands for the retardance of the beam splitter
    phi = float(rng.uniform(-np.pi, np.pi))
    e_pol = field(test, t_now)[:, 1].numpy().copy()
    e_pol[:, 0] *= np.sqrt(1 + b_true) * np.exp(1j * phi)
    e_pol[:, 1] *= np.sqrt(1 - b_true)

    yy, xx = np.ogrid[:n, :n]
    mask = (yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2 <= (n / 2.0 / 1.1) ** 2
    # true delta and how diagonal the relation is
    cx = np.sum(e_pol[:, 0] * e_cam[:, 0].conj()) / np.sum(np.abs(e_cam[:, 0]) ** 2)
    cy = np.sum(e_pol[:, 1] * e_cam[:, 1].conj()) / np.sum(np.abs(e_cam[:, 1]) ** 2)
    delta_true = float(np.degrees(np.angle(cx) - np.angle(cy)))
    cross = np.sum(e_pol[:, 0] * e_cam[:, 1].conj()) / np.sqrt(np.sum(np.abs(e_cam[:, 1]) ** 2) * np.sum(np.abs(e_pol[:, 0]) ** 2))
    fid_x = abs(np.sum(e_pol[:, 0] * e_cam[:, 0].conj())) / np.sqrt(np.sum(np.abs(e_cam[:, 0]) ** 2) * np.sum(np.abs(e_pol[:, 0]) ** 2))
    out.update({"delta_true_deg": float(wrap_deg(delta_true)), "cross_term_amplitude": float(abs(cross)), "field_fidelity_x": float(fid_x)})

    # agreement between the model and the true image at 45 deg, with the true delta
    e_ref = e_cam.copy(); e_ref[:, 0] = e_ref[:, 0] * np.exp(1j * np.radians(delta_true))
    out["pcc_image_45deg"] = float(np.mean([pcc(polarizer_image(e_ref[k], 45.0)[mask], polarizer_image(e_pol[k], 45.0)[mask])
                                            for k in range(n_patterns)]))
    res = {}
    for noise in noises:
        i45 = np.stack([polarizer_image(e_pol[k], 45.0) for k in range(n_patterns)])
        i135 = np.stack([polarizer_image(e_pol[k], 135.0) for k in range(n_patterns)])
        i48 = np.stack([polarizer_image(e_pol[k], 48.0) for k in range(n_patterns)])
        sig = noise * i45[:, mask].mean()
        i45 = i45 + sig * rng.normal(size=i45.shape)
        i135 = i135 + sig * rng.normal(size=i135.shape)
        i48 = i48 + sig * rng.normal(size=i48.shape)
        est = {}
        for kk in (1, 4, 16):
            if kk > n_patterns:
                continue
            groups = [slice(g0, g0 + kk) for g0 in range(0, n_patterns - kk + 1, kk)]
            est[f"one_angle_{kk}_images"] = [estimate_delta_one_angle(i45[g], e_cam[g], mask) for g in groups]
            est[f"two_angles_{kk}_patterns"] = [estimate_delta_two_angles(i45[g], i135[g], e_cam[g], mask) for g in groups]
        one = [slice(k, k + 1) for k in range(n_patterns)]
        est["one_image_at_48deg_free_terms"] = [estimate_delta_one_angle(i48[g], e_cam[g], mask) for g in one]
        est["one_image_at_48deg_nominal_terms"] = [estimate_delta_one_angle(i48[g], e_cam[g], mask, free_axis_terms=False) for g in one]
        res[str(noise)] = {name: np.abs(wrap_deg(np.array(v) - delta_true)).tolist() for name, v in est.items()}
    out["abs_error_deg"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--b", type=float, default=0.70)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--data-noise", type=float, default=0.0, help="multiplicative noise of the retrieval data")
    ap.add_argument("--mu", type=float, default=1.0, help="field correlation of the fiber between retrieval and polarizer image")
    ap.add_argument("--noises", type=str, default="0,0.05,0.2", help="noise of the polarizer image, fraction of its mean")
    ap.add_argument("--patterns", type=int, default=16)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    noises = [float(v) for v in a.noises.split(",")]
    res = []
    for sd in [int(x) for x in a.seeds.split(",")]:
        r = run(sd, a.b, a.grid, a.budget, a.data_noise, a.mu, noises, a.patterns)
        res.append(r)
        print(f"seed {sd}: M={r['modes']} delta_true {r['delta_true_deg']:+.1f} deg, cross term {r['cross_term_amplitude']:.4f}, "
              f"fidelity {r['field_fidelity_x']:.4f}, PCC of the 45 deg image {r['pcc_image_45deg']:.3f}", flush=True)
    print(f"\nmean PCC of the 45 deg image: {np.mean([r['pcc_image_45deg'] for r in res]):.3f}")
    print("absolute error of delta in degrees: median / 90th percentile / maximum over fibers and SLM patterns")
    summary = {}
    for noise in noises:
        summary[str(noise)] = {}
        for name in res[0]["abs_error_deg"][str(noise)]:
            v = np.concatenate([r["abs_error_deg"][str(noise)][name] for r in res])
            summary[str(noise)][name] = [float(np.median(v)), float(np.percentile(v, 90)), float(v.max())]
            print(f"  image noise {noise:4.2f}  {name:34s} {np.median(v):6.2f} / {np.percentile(v, 90):6.2f} / {v.max():6.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "summary": summary, "runs": res}, indent=1))


if __name__ == "__main__":
    main()
