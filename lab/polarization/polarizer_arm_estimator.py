"""
Closed-form polarization frame when the arm that holds the polarizer has a
polarization-dependent transmission (for example a pellicle beam splitter in front of it).

Model of the field in front of the polarizer:

    E_pol = c * D * U * E_model,      D = diag(sqrt(1 + d), sqrt(1 - d)),

U: unitary Jones matrix between the TM frame and the lab frame (three unknowns);
d: diattenuation of the arm relative to the TM, -1 < d < 1, positive when lab x is passed
   better (one unknown);
lab x, y: the polarizer at stage angle theta0 and theta0 + 90 deg.

Two estimators, both closed form:

A. FROM FOCI (recommended; robust to errors of the TM in predicting speckle)
   Three two-foci patterns are designed in the TM's own frame with the matrices
   PROBE_DESIGNS (bases x'/y', +45'/-45', R'/L'). Each puts two orthogonal, known model
   states on two spots. The polarizer records each pattern at lab x and lab y only.
   For every focus, p = (P_x - P_y) / (P_x + P_y). With u = R_1 . s the lab S1 of that state
   before the arm:  p = (d + u) / (1 + d u).  An orthogonal pair has u_B = -u_A, which gives
   d; then the three u are the components of R_1, the model state that reaches the lab as x.
   -> frame_from_focus_pairs()

B. FROM SPECKLE (cross-check; needs polarizer images at >= 3 angles)
   With A1 = I_0 - I_90 and A2 = I_45 - I_135 and the model Stokes images (S0, s):
       A1 = g d S0 + g R_1 . s,     A2 = g sqrt(1 - d^2) R_2 . s
   so the S0 coefficient gives d (and the axis theta0), and the s coefficients give two rows
   of the rotation R. A constant per image is fitted too, so camera offsets drop out.
   -> estimate_frame_with_diattenuation()

Run "python polarizer_arm_estimator.py" for a self-test. Needs numpy only.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

SIGMA = (np.array([[1, 0], [0, -1]], dtype=complex),      # acts on S1
         np.array([[0, 1], [1, 0]], dtype=complex),       # S2
         np.array([[0, -1j], [1j, 0]], dtype=complex))    # S3


def stokes(e_xy: np.ndarray) -> np.ndarray:
    """(..., 2, H, W) complex field -> (4, ..., H, W) real Stokes parameters S0..S3."""
    ex, ey = e_xy[..., 0, :, :], e_xy[..., 1, :, :]
    c = ex * np.conj(ey)
    return np.stack([np.abs(ex) ** 2 + np.abs(ey) ** 2, np.abs(ex) ** 2 - np.abs(ey) ** 2, 2 * c.real, -2 * c.imag])


def mueller_rotation(u: np.ndarray) -> np.ndarray:
    return np.array([[0.5 * np.trace(SIGMA[i] @ u @ SIGMA[k] @ u.conj().T).real for k in range(3)] for i in range(3)])


def jones_from_rotation(r: np.ndarray) -> np.ndarray:
    """SU(2) Jones matrix (up to sign) whose Poincare-sphere rotation is r."""
    phi = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    if phi < 1e-9:
        return np.eye(2, dtype=complex)
    if np.pi - phi < 1e-6:
        m = (r + np.eye(3)) / 2.0
        n = m[:, int(np.argmax(np.diag(m)))]
        n = n / np.linalg.norm(n)
    else:
        n = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * np.sin(phi))
    return np.cos(phi / 2) * np.eye(2) - 1j * np.sin(phi / 2) * (n[0] * SIGMA[0] + n[1] * SIGMA[1] + n[2] * SIGMA[2])


def lab_stokes_from_polarizer_stack(images: np.ndarray, angles_deg: Sequence[float]) -> np.ndarray:
    """Per-pixel least squares of I_a = (A0 + A1 cos 2a + A2 sin 2a) / 2. images: (n_angles, ..., H, W) -> (3, ..., H, W)."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    design = 0.5 * np.stack([np.ones_like(a), np.cos(2 * a), np.sin(2 * a)], axis=1)
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Need at least three polarizer angles that are distinct modulo 180 deg.")
    return np.tensordot(np.linalg.pinv(design), np.asarray(images, dtype=np.float64), axes=(1, 0))


def estimate_frame_with_diattenuation(images: np.ndarray, angles_deg: Sequence[float], e_model: np.ndarray,
                                      mask: np.ndarray, fit_axis: bool = True, min_diattenuation_for_axis: float = 0.05) -> Dict:
    """
    images:  (n_angles, K, H, W) registered polarizer images of K SLM patterns (stage angles angles_deg).
    e_model: (K, 2, H, W) model field of the same patterns, in the plane of the camera behind the polarizer.
    mask:    (H, W) bool, pixels to use.
    fit_axis: also find the stage angle theta0 of the diattenuation axis (else theta0 = 0).

    Returns U, R, d, theta0_deg and quality figures. The lab frame of U has its x axis at stage angle theta0.
    """
    lab = lab_stokes_from_polarizer_stack(images, angles_deg)
    a0, a1, a2 = (lab[i][..., mask].ravel() for i in range(3))
    s = stokes(e_model)
    s0 = s[0][..., mask].ravel()
    sm = np.stack([s[k][..., mask].ravel() for k in (1, 2, 3)], axis=1)          # (n, 3)
    x4 = np.column_stack([s0, sm])                                               # (n, 4)
    x5 = np.column_stack([np.ones_like(s0), x4])                                 # with a constant: absorbs camera offsets

    def fit(y):
        return np.linalg.lstsq(x5, y, rcond=None)[0][1:]

    b1, b2 = fit(a1), fit(a2)
    # axis of the diattenuation from the S0 coefficients: they point along (cos 2 phi, sin 2 phi) of the favoured axis
    strength = float(np.hypot(b1[0], b2[0]) / max(np.linalg.norm(b1[1:]), 1e-30))
    theta0 = 0.0
    if fit_axis and strength >= min_diattenuation_for_axis:
        fav = 0.5 * np.arctan2(b2[0], b1[0])                                     # favoured axis, (-90, 90] deg
        theta0 = fav if abs(fav) <= np.pi / 4 else fav - np.sign(fav) * np.pi / 2  # diattenuation axis nearest to stage 0
    c2, s2 = np.cos(2 * theta0), np.sin(2 * theta0)
    a1r, a2r = c2 * a1 + s2 * a2, -s2 * a1 + c2 * a2                             # lab Stokes in axes rotated by theta0
    b1, b2 = fit(a1r), fit(a2r)
    beta0, beta, gamma0, gamma = b1[0], b1[1:], b2[0], b2[1:]
    nb, ng = float(np.linalg.norm(beta)), float(np.linalg.norm(gamma))
    d = float(np.clip(beta0 / nb, -0.98, 0.98))
    q = np.stack([beta / nb, gamma / ng])
    w, _, vt = np.linalg.svd(q, full_matrices=False)
    q = w @ vt
    r = np.vstack([q, np.cross(q[0], q[1])])
    fit1 = nb * (d * s0 + sm @ q[0])
    fit2 = nb * np.sqrt(1.0 - d ** 2) * (sm @ q[1])
    explained = 1.0 - float(((a1r - fit1) ** 2).sum() + ((a2r - fit2) ** 2).sum()) / float((a1r ** 2).sum() + (a2r ** 2).sum())
    # independent estimates of |d| and d, as consistency checks
    ratio = ng / nb                                                              # = sqrt(1 - d^2)
    d_from_ratio = float(np.sqrt(max(0.0, 1.0 - min(ratio, 1.0) ** 2)))
    b0 = np.linalg.lstsq(x5, a0, rcond=None)[0]                                  # A0 = offset + alpha0 S0 + alpha . s
    d_from_total = float((b0[2:] @ q[0]) / b0[1])
    return {
        "U": jones_from_rotation(r), "R": r, "d": d, "theta0_deg": float(np.degrees(theta0)),
        "weights_xy": [1.0 + d, 1.0 - d], "gain": nb, "explained_fraction": explained,
        "d_from_s2_amplitude": d_from_ratio, "d_from_total_intensity": d_from_total,
        "s0_leak_into_a2": float(gamma0 / nb),                                   # ~0 if theta0 is right
        "xprime_azimuth_deg": float(np.degrees(0.5 * np.arctan2(r[1, 0], r[0, 0])) % 180.0),
        "xprime_ellipticity_deg": float(np.degrees(0.5 * np.arcsin(np.clip(r[2, 0], -1.0, 1.0)))),
    }


def field_at_polarizer(e_model: np.ndarray, est: Dict) -> np.ndarray:
    """(..., 2, H, W) model field -> field in front of the polarizer, in the lab frame of ``est`` (x at stage angle theta0)."""
    u, d = est["U"], est["d"]
    ex, ey = e_model[..., 0, :, :], e_model[..., 1, :, :]
    return np.stack([np.sqrt(1 + d) * (u[0, 0] * ex + u[0, 1] * ey), np.sqrt(1 - d) * (u[1, 0] * ex + u[1, 1] * ey)], axis=-3)


def predict_polarizer_image(e_model: np.ndarray, est: Dict, stage_angle_deg: float) -> np.ndarray:
    e = field_at_polarizer(e_model, est)
    a = np.radians(stage_angle_deg - est["theta0_deg"])
    return np.abs(np.cos(a) * e[..., 0, :, :] + np.sin(a) * e[..., 1, :, :]) ** 2


def pass_fraction_before_arm(p_x: float, p_y: float, est: Dict) -> float:
    """Fraction of a focus polarized along lab x BEFORE the arm, from the powers measured behind the polarizer at x and y."""
    wx, wy = est["weights_xy"]
    return (p_x / wx) / (p_x / wx + p_y / wy)


# ---------------------------------------------------------------------------------------
# A. frame from foci
# ---------------------------------------------------------------------------------------
PROBE_ROTATIONS = {"x'/y'": np.eye(3),
                   "+45'/-45'": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                   "R'/L'": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
# Design matrix of each probe pattern: use it in place of U in the two-foci optimization.
# Focus A then has the model Stokes vector +e_k and focus B has -e_k (k = 1, 2, 3).
PROBE_DESIGNS = {name: jones_from_rotation(r) for name, r in PROBE_ROTATIONS.items()}


def diattenuation_from_orthogonal_pair(p_a: float, p_b: float) -> float:
    """d from two foci in orthogonal states: (p_a - d)/(1 - d p_a) = -(p_b - d)/(1 - d p_b), root with |d| < 1."""
    ssum, prod = p_a + p_b, 1.0 + p_a * p_b
    if abs(ssum) < 1e-9:
        return 0.0
    return float((prod - np.sqrt(max(prod ** 2 - ssum ** 2, 0.0))) / ssum)


def rotation_with_first_row(r1: np.ndarray) -> np.ndarray:
    """A proper rotation whose first row is r1 (normalized); the other two rows are arbitrary."""
    r1 = np.asarray(r1, dtype=float) / np.linalg.norm(r1)
    r2 = np.cross(r1, np.eye(3)[int(np.argmin(np.abs(r1)))])
    r2 = r2 / np.linalg.norm(r2)
    return np.vstack([r1, r2, np.cross(r1, r2)])


def frame_from_focus_pairs(powers: Dict[str, Sequence[float]]) -> Dict:
    """
    powers: {basis name: (P_A_x, P_A_y, P_B_x, P_B_y)} for the three bases of PROBE_DESIGNS, in
    that order of keys or by name. P_.._x / P_.._y: power of the focus (sum over its window, dark
    level removed, no speckle background removed) with the polarizer at lab x / lab y.

    Returns d, R_1, the design matrix U for the final two foci (A along lab x, B along lab y),
    and consistency figures: |R_1| before normalization (1 for pure, orthogonal foci) and, per
    basis, u_A + u_B (0 for an orthogonal pair).
    """
    names = list(PROBE_ROTATIONS)
    p = np.array([[(powers[n][0] - powers[n][1]) / (powers[n][0] + powers[n][1]),
                   (powers[n][2] - powers[n][3]) / (powers[n][2] + powers[n][3])] for n in names])
    # A focus that reads exactly 0 counts at one polarizer angle gives p = +-1. Such a pair does not
    # constrain d and made the expressions below 0/0 (lab run of 15 Nov 2026). Keep p inside (-1, 1)
    # and weight each pair by how far both of its foci are from full extinction.
    p = np.clip(p, -1.0 + 1e-6, 1.0 - 1e-6)
    d_k = np.clip([diattenuation_from_orthogonal_pair(pa, pb) for pa, pb in p], -0.98, 0.98)
    u_k = np.array([0.5 * ((pa - dk) / (1 - dk * pa) - (pb - dk) / (1 - dk * pb)) for (pa, pb), dk in zip(p, d_k)])
    weights = np.clip(1.0 - u_k ** 2, 0.0, None) * np.sqrt((1.0 - p[:, 0] ** 2) * (1.0 - p[:, 1] ** 2))
    d_defined = bool(weights.sum() > 1e-6)
    d = float(np.clip((weights * d_k).sum() / weights.sum(), -0.98, 0.98)) if d_defined else 0.0
    u_a = (p[:, 0] - d) / (1 - d * p[:, 0])
    u_b = (p[:, 1] - d) / (1 - d * p[:, 1])
    r1 = 0.5 * (u_a - u_b)
    r = rotation_with_first_row(r1)
    return {"d": d, "d_defined": d_defined, "d_per_basis": d_k.tolist(), "weights_xy": [1.0 + d, 1.0 - d],
            "R1": (r1 / np.linalg.norm(r1)).tolist(), "R1_norm": float(np.linalg.norm(r1)),
            "pair_imbalance": (u_a + u_b).tolist(), "p": p.tolist(), "R": r, "U": jones_from_rotation(r)}


def _self_test_foci() -> None:
    rng = np.random.default_rng(7)
    for d_true in (-0.67, 0.29, 0.0):
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        u_true = q / np.sqrt(np.linalg.det(q))
        powers = {}
        for name, design in PROBE_DESIGNS.items():
            vals = []
            for channel in (0, 1):                          # focus A: design channel 0, focus B: channel 1
                jones_model = design.conj().T[:, channel]   # model state that the design sends to that channel
                e = u_true @ jones_model * rng.uniform(0.7, 1.3)
                px, py = (1 + d_true) * abs(e[0]) ** 2, (1 - d_true) * abs(e[1]) ** 2
                vals += [px * (1 + 0.01 * rng.normal()), py * (1 + 0.01 * rng.normal())]
            powers[name] = vals
        est = frame_from_focus_pairs(powers)
        r1_true = mueller_rotation(u_true)[0]
        err = np.degrees(np.arccos(np.clip(np.dot(est["R1"], r1_true), -1, 1)))
        # the design U must send the model state R_1 to lab x: first row of its rotation = R_1
        assert np.allclose(mueller_rotation(est["U"])[0], est["R1"], atol=1e-9)
        print(f"foci: d true {d_true:+.2f} -> {est['d']:+.3f} | |R1| {est['R1_norm']:.3f} | angle between R1 and truth {err:.2f} deg (Poincare sphere)")
        assert abs(est["d"] - d_true) < 0.03 and err < 3.0 and abs(est["R1_norm"] - 1) < 0.05
    # a focus that reads exactly 0 counts at one angle (x'/y' pair on the arm axes) must not give NaN
    d_true = -0.7
    powers = {"x'/y'": [0.0, 500.0, 300.0, 0.0]}
    for name, (sa, sb) in (("+45'/-45'", (0.0, 0.0)), ("R'/L'", (0.0, 0.0))):      # lab S1 = 0 for both foci
        powers[name] = [(1 + d_true) * 100.0, (1 - d_true) * 100.0, (1 + d_true) * 80.0, (1 - d_true) * 80.0]
    est = frame_from_focus_pairs(powers)
    assert np.all(np.isfinite(est["R1"])) and np.isfinite(est["d"]) and abs(est["d"] - d_true) < 0.01, est
    assert abs(abs(est["R1"][0]) - 1.0) < 1e-3
    print(f"foci with a fully blocked focus: d {est['d']:+.3f}, R1 {np.round(est['R1'], 3).tolist()}")
    # numbers of the lab run of 12 Nov (uncorrected pattern = basis x'/y'): d = -0.72
    assert abs(diattenuation_from_orthogonal_pair((195.75 - 1026.16) / (195.75 + 1026.16), (124.89 - 887.69) / (124.89 + 887.69)) + 0.718) < 2e-3


def _self_test() -> None:
    _self_test_foci()
    rng = np.random.default_rng(3)
    n, k = 64, 4
    f = np.fft.fftfreq(n)
    pupil = np.hypot(*np.meshgrid(f, f, indexing="ij")) <= 0.15
    yy, xx = np.ogrid[:n, :n]
    core = (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= (n / 2.2) ** 2
    e_model = np.fft.ifft2((rng.normal(size=(k, 2, n, n)) + 1j * rng.normal(size=(k, 2, n, n))) * pupil) * core
    e_model[:, 1] *= 0.6                                    # the model field is partly polarized on average
    angles = [0.0, 45.0, 90.0, 135.0]
    for d_true, th_true in ((-0.5, 0.0), (0.3, 4.0), (-0.3, -6.0), (0.0, 0.0)):
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        u_true = q / np.sqrt(np.linalg.det(q))
        ex = u_true[0, 0] * e_model[:, 0] + u_true[0, 1] * e_model[:, 1]
        ey = u_true[1, 0] * e_model[:, 0] + u_true[1, 1] * e_model[:, 1]
        ex, ey = np.sqrt(1 + d_true) * ex, np.sqrt(1 - d_true) * ey
        imgs = []
        for a in angles:
            ar = np.radians(a - th_true)
            clean = np.abs(np.cos(ar) * ex + np.sin(ar) * ey) ** 2
            imgs.append(clean + 0.03 * clean.mean() * rng.normal(size=clean.shape) + 0.01)   # noise + camera offset
        est = estimate_frame_with_diattenuation(np.stack(imgs), angles, e_model, core)
        err = np.degrees(np.arccos(np.clip((np.trace(est["R"] @ mueller_rotation(u_true).T) - 1) / 2, -1, 1))) / 2
        print(f"d true {d_true:+.2f} -> {est['d']:+.3f} (from S2 amplitude {est['d_from_s2_amplitude']:.3f}, from total {est['d_from_total_intensity']:+.3f}) | "
              f"axis true {th_true:+.1f} -> {est['theta0_deg']:+.2f} deg | frame error {err:.2f} deg | explained {est['explained_fraction']:.3f}")
        assert abs(est["d"] - d_true) < 0.03 and err < 1.0 and abs(est["theta0_deg"] - th_true) < 1.0
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
