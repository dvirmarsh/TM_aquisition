"""
Find the polarization frame of a retrieved TM with a rotating polarizer, then form two foci
in the two linear laboratory polarizations.

Place this file in TM_lib (next to run_tm_focus_optimization.py) and run it on the lab PC:

    python run_polarization_axes.py --self-test      # math only, no hardware
    python run_polarization_axes.py --test-stage     # only moves the stage through the angles
    python run_polarization_axes.py                  # full lab run (polarizer in front of the Fourier-plane camera)
    python run_polarization_axes.py --plane image    # same, polarizer in front of the image-plane camera
    python run_polarization_axes.py --manual-stage   # rotate the polarizer by hand when prompted

``RotatingPolarizerStage`` below drives the lab's Thorlabs KDC101 rotation mount (serial 27268833).
``--plane`` names the transmission camera that sits behind the polarizer: "fourier" (default,
the lab setup of 12 Nov 2026) or "image". The algorithm is the same for both planes.

What the lab run does
---------------------
1. Loads the TM checkpoint written by run_tm_focus_optimization.py (retrieved WITHOUT a polarizer).
2. Displays a few speckle SLM patterns and records the camera behind the polarizer
   at a few stage angles (default 0, 45, 90, 135 deg), all at one exposure.
3. Registers every polarizer angle to the model (the polarizer displaces the image, and the
   displacement changes as it rotates), optionally fits a low-order aberration added by the
   polarizer, and solves in closed form for the global Jones matrix U with
   E_lab = c * U * E_model.
4. Optimizes one SLM pattern for two foci: focus A polarized along lab x, focus B along lab y.
5. Records that pattern behind the polarizer at the x and y angles, saves the images, the SLM
   pattern and a JSON summary.

Definitions
-----------
* "Lab x" is the transmission axis of the polarizer at stage angle ``polarizer_x_angle_deg``;
  "lab y" is that angle + 90 deg. Only relative stage angles matter. The frame is that of the
  field arriving at the polarizer (it includes the beam splitter in front of it).
* The direction in which the stage turns (clockwise or not, as seen by the camera) does not
  have to be known: the solution is expressed in stage angles.
* Stokes parameters of a field (E_x, E_y): S0 = |Ex|^2 + |Ey|^2, S1 = |Ex|^2 - |Ey|^2,
  S2 = 2 Re(Ex Ey*), S3 = -2 Im(Ex Ey*). A polarizer at angle a transmits
  I_a = (S0 + S1 cos 2a + S2 sin 2a) / 2.
* All image coordinates are (y, x) pixels of the n_out x n_out model grid, in the orientation of
  the TM model (camera frame after crop and resize; the image-plane frame is also np.flip-ed, the
  Fourier-plane frame is not, as in run_tm_focus_optimization.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

TM_LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TM_LIB_DIR.parent
for _p in (PROJECT_ROOT, TM_LIB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# =====================================================================================
# ROTATION STAGE -- Thorlabs KDC101 through the lab APT wrapper
# =====================================================================================
class RotatingPolarizerStage:
    """
    Motorized rotation stage that holds the linear polarizer in front of the camera of ``--plane``.

    Requirements that this class meets:
      * ``_move_absolute_blocking(angle_deg)`` moves to an ABSOLUTE stage angle in degrees and
        returns only when the stage has stopped;
      * any real angle may be requested; it is reduced modulo 360;
      * the same stage angle always gives the same polarizer orientation (``connect`` homes the
        stage if the controller has lost its reference at power-up).

    Lab PC implementation (12 Nov 2026): Thorlabs KDC101 K-Cube, serial 27268833, set up in APT as a
    rotation stage (degrees, 0-360, PRM1-Z8 gearing). It is driven through the APT wrapper that the
    lab code already uses (PROJECT_ROOT/thorlabs_apt, imported by create_refs.py), so APT.dll is
    initialised only once per process. The stage is addressed by serial number, never by list index.
    """

    SERIAL_NUMBER = 27268833             # KDC101 that drives the polarizer mount
    POSITION_TOLERANCE_DEG = 0.2         # a move fails if the read-back angle is further than this from the request
    STOP_TIMEOUT_S = 10.0                # wait this long for the "in motion" status bits to clear after a move
    APT_STAGE_UNITS_DEG = 2              # MOT_GetStageAxisInfo: 1 = mm, 2 = degrees

    def __init__(self, settle_s: float = 0.3, serial_number: int = SERIAL_NUMBER, home_if_needed: bool = True):
        self.settle_s = float(settle_s)      # wait after each move, for vibrations to die out
        self.serial_number = int(serial_number)
        self.home_if_needed = bool(home_if_needed)
        self._device = None

    def _wait_until_stopped(self) -> None:
        deadline = time.time() + self.STOP_TIMEOUT_S
        while self._device.is_in_motion:
            if time.time() > deadline:
                raise RuntimeError(f"Rotation stage {self.serial_number} still reports motion {self.STOP_TIMEOUT_S:g} s after the move.")
            time.sleep(0.05)

    def connect(self) -> None:
        import thorlabs_apt.thorlabs_apt as apt      # same module object as in create_refs.py

        serials = [int(sn) for _hw_type, sn in apt.list_available_devices()]
        if self.serial_number not in serials:
            raise RuntimeError(f"Rotation stage controller {self.serial_number} not found by APT (found {serials}). "
                               "Check the USB cable and close Kinesis / APT User, which block the controller.")
        motor = apt.Motor(self.serial_number)
        _min_pos, _max_pos, units, _pitch = motor.get_stage_axis_info()
        if int(units) != self.APT_STAGE_UNITS_DEG:
            raise RuntimeError(f"Controller {self.serial_number} is not set up as a rotation stage (APT stage units {units}, expected degrees).")
        self._device = motor
        if not motor.is_channel_enabled:
            motor.enable()
        # The controller loses its angle reference at power-up. Home once, so that the same stage
        # angle always gives the same polarizer orientation; later connections skip this.
        if not motor.has_homing_been_completed:
            if not self.home_if_needed:
                raise RuntimeError(f"Rotation stage {self.serial_number} is not homed and home_if_needed is False.")
            print(f"Rotation stage {self.serial_number}: not homed since power-up; homing (up to one full turn) ...")
            motor.move_home(True)
            self._wait_until_stopped()
            if not motor.has_homing_been_completed:
                raise RuntimeError(f"Rotation stage {self.serial_number}: homing did not complete.")
        print(f"Rotation stage {self.serial_number} connected, homed, at {motor.position:.3f} deg")

    def _move_absolute_blocking(self, angle_deg: float) -> None:
        if self._device is None:
            raise RuntimeError("Rotation stage is not connected; call connect() first.")
        target = float(angle_deg) % 360.0            # APT accepts 0..360 for this stage
        self._device.move_to(target, blocking=True)  # returns when APT reports the move complete
        self._wait_until_stopped()
        reached = float(self._device.position)
        error = abs((reached - target + 180.0) % 360.0 - 180.0)
        if error > self.POSITION_TOLERANCE_DEG or self._device.motion_error:
            raise RuntimeError(f"Rotation stage {self.serial_number}: requested {target:.3f} deg, reached {reached:.3f} deg "
                               f"(error {error:.3f} deg, motion_error {self._device.motion_error}).")

    def position(self) -> float:
        if self._device is None:
            raise RuntimeError("Rotation stage is not connected; call connect() first.")
        return float(self._device.position)

    def close(self) -> None:
        # The APT wrapper has no per-device close: APTCleanUp runs once, at interpreter exit.
        self._device = None

    def move_to(self, angle_deg: float) -> None:
        self._move_absolute_blocking(float(angle_deg))
        time.sleep(self.settle_s)


class ManualPolarizerStage:
    """Fallback without a motor: asks the operator to set each angle by hand (--manual-stage)."""

    def __init__(self) -> None:
        self._angle = 0.0

    def connect(self) -> None:
        print("Manual polarizer stage: you will be asked to set each angle by hand.")

    def move_to(self, angle_deg: float) -> None:
        input(f"  >> Set the polarizer to {angle_deg:.1f} deg, then press Enter ")
        self._angle = float(angle_deg)

    def position(self) -> float:
        return self._angle

    def close(self) -> None:
        pass


# =====================================================================================
# Configuration
# =====================================================================================
# Camera plane behind the polarizer. Same values as ``plane_configs`` in run_tm_focus_optimization.run():
# index of the plane in the model output, index in pipeline.cameras, crop name, whether the cropped
# frame is np.flip-ed to reach the model orientation, and the PipelineConfig field with the exposure.
PLANES = {
    "image": {"index": 0, "camera": 0, "crop": "transmission_image", "flip": True, "exposure_field": "transmission_image_et"},
    "fourier": {"index": 1, "camera": 1, "crop": "transmission_fourier", "flip": False, "exposure_field": "transmission_fourier_et"},
}


@dataclass
class PolarizationConfig:
    # --- camera plane behind the polarizer: "fourier" or "image" (transmission side)
    plane: str = "fourier"
    # --- polarizer / stage
    polarizer_x_angle_deg: float = 0.0          # stage angle that defines lab x
    frame_angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)   # relative to lab x; >= 3 distinct mod 180
    verify_angles_deg: Tuple[float, ...] = (0.0, 90.0)               # 0 = lab x, 90 = lab y
    # --- speckle patterns used to find the frame
    n_speckle_patterns: int = 4
    speckle_exposure: float = 0.0               # 0 -> transmission_<plane>_et[0] * gain of PipelineConfig
    random_seed: int = 0
    # --- displacement and aberration caused by the polarizer
    max_shift_px: float = 12.0                  # largest image displacement searched for (model-grid pixels)
    fit_aberrations: bool = True                # fit a low-order pupil phase added by the polarizer
    n_zernike: int = 3                          # 3: defocus + astigmatism; 6: + coma + spherical
    aberration_min_gain: float = 0.01           # keep the fit only if the mean PCC rises by this much
    compensate_aberration_in_design: bool = True   # if the fit is kept, design the foci to be sharp on the camera
    # --- two foci, (y, x) on the model grid; None -> defaults around the centre
    target_x_pol_yx: Optional[Tuple[int, int]] = None    # focus A, polarized along lab x
    target_y_pol_yx: Optional[Tuple[int, int]] = None    # focus B, polarized along lab y
    window_half_width: int = 1                  # focus window is (2h+1) x (2h+1) pixels
    slm_opt_iters: int = 150
    slm_opt_lr: float = 0.1
    cross_pol_penalty: bool = True              # also suppress the cross-polarized light at each focus
    foci_exposure: float = 24.0                 # start value; halved automatically on saturation
    also_uncorrected: bool = False              # also record foci designed in the TM's own x'/y' axes
    # --- files
    checkpoint_path: str = ""                   # "" -> PipelineConfig.tm_checkpoint_path
    output_subdir: str = ""                     # inside PipelineConfig.output_dir; "" -> polarization_axes (image) or polarization_axes_fourier

    def __post_init__(self):
        self.plane = str(self.plane).lower()
        if self.plane not in PLANES:
            raise ValueError(f"plane must be one of {sorted(PLANES)}, got {self.plane!r}")
        if not self.output_subdir:
            self.output_subdir = "polarization_axes" if self.plane == "image" else f"polarization_axes_{self.plane}"


# =====================================================================================
# Polarization algebra (numpy)
# =====================================================================================
SIGMA = (np.array([[1, 0], [0, -1]], dtype=complex),      # acts on S1
         np.array([[0, 1], [1, 0]], dtype=complex),       # S2
         np.array([[0, -1j], [1j, 0]], dtype=complex))    # S3


def stokes(e_xy: np.ndarray) -> np.ndarray:
    """(..., 2, H, W) complex field -> (4, ..., H, W) real Stokes parameters S0..S3."""
    ex, ey = e_xy[..., 0, :, :], e_xy[..., 1, :, :]
    c = ex * np.conj(ey)
    return np.stack([np.abs(ex) ** 2 + np.abs(ey) ** 2, np.abs(ex) ** 2 - np.abs(ey) ** 2, 2 * c.real, -2 * c.imag])


def polarizer_intensity(e_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.abs(np.cos(a) * e_xy[..., 0, :, :] + np.sin(a) * e_xy[..., 1, :, :]) ** 2


def apply_jones(u: np.ndarray, e_xy: np.ndarray) -> np.ndarray:
    """E -> U E; the last three axes of e_xy are (2, H, W)."""
    ex, ey = e_xy[..., 0, :, :], e_xy[..., 1, :, :]
    return np.stack([u[0, 0] * ex + u[0, 1] * ey, u[1, 0] * ex + u[1, 1] * ey], axis=-3)


def mueller_rotation(u: np.ndarray) -> np.ndarray:
    """SO(3) rotation of (S1, S2, S3) produced by a unitary Jones matrix."""
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
    """
    Per-pixel least squares of I_a = (A0 + A1 cos 2a + A2 sin 2a) / 2 over the polarizer angles.
    images: (n_angles, ..., H, W). Returns (3, ..., H, W) = gain * (S0, S1, S2) in the lab frame.
    For 0/45/90/135 deg this is A1 = I_0 - I_90 and A2 = I_45 - I_135.
    """
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    design = 0.5 * np.stack([np.ones_like(a), np.cos(2 * a), np.sin(2 * a)], axis=1)      # (n_angles, 3)
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Need at least three polarizer angles that are distinct modulo 180 deg.")
    return np.tensordot(np.linalg.pinv(design), np.asarray(images, dtype=np.float64), axes=(1, 0))


def estimate_frame(images: np.ndarray, angles_deg: Sequence[float], e_model: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Closed-form global Jones matrix U with E_lab = c U E_model.

    images: (n_angles, K, H, W) registered polarizer images of K SLM patterns.
    e_model: (K, 2, H, W) model field of the same patterns. mask: (H, W) bool, pixels to use.
    Least squares A_i(r) = g R_i . s(r), i = 1, 2, with s = (S1, S2, S3) of the model:
    a_i = (sum s s^T)^-1 sum A_i s; rows made orthonormal by an SVD; third row = cross product.
    A constant camera offset drops out (it only enters A0, which is not used).
    """
    lab = lab_stokes_from_polarizer_stack(images, angles_deg)                # (3, K, H, W)
    s = stokes(e_model)                                                      # (4, K, H, W)
    sm = np.stack([s[k][..., mask].ravel() for k in (1, 2, 3)], axis=1)      # (n, 3)
    gram = sm.T @ sm
    a = np.stack([np.linalg.solve(gram, sm.T @ lab[i][..., mask].ravel()) for i in (1, 2)])   # (2, 3) = g [R1; R2]
    w, sv, vt = np.linalg.svd(a, full_matrices=False)
    q = w @ vt
    r = np.vstack([q, np.cross(q[0], q[1])])
    gain = float(sv.mean())
    meas = np.stack([lab[i][..., mask].ravel() for i in (1, 2)], axis=1)
    explained = 1.0 - float(((meas - sm @ (gain * q).T) ** 2).sum() / (meas ** 2).sum())
    return {
        "U": jones_from_rotation(r), "R": r, "gain": gain,
        "explained_fraction": explained,                                     # 1 = polarizer data fully explained
        "singular_values": [float(v) for v in sv],                           # equal for a clean fit
        # Model x' axis as seen in the lab: azimuth and ellipticity of its polarization ellipse.
        "xprime_azimuth_deg": float(np.degrees(0.5 * np.arctan2(r[1, 0], r[0, 0])) % 180.0),
        "xprime_ellipticity_deg": float(np.degrees(0.5 * np.arcsin(np.clip(r[2, 0], -1.0, 1.0)))),
        # If U were only a rotation of the axes: its angle, and how far the data are from that case.
        "rotation_only_angle_deg": float(np.degrees(0.5 * np.arctan2(r[1, 0] - r[0, 1], r[0, 0] + r[1, 1])) % 180.0),
        "out_of_plane": float(np.hypot(q[0, 2], q[1, 2])),                   # 0 if U is a pure rotation of the axes
        "separation_without_correction": float(np.hypot(r[0, 0], r[1, 0])),  # best case when only the polarizer is rotated
    }


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def peak_to_background_ratio(intensity: np.ndarray, support: np.ndarray, target_yx: Tuple[int, int],
                             search_radius: int = 3, half_width: int = 1) -> Tuple[float, Tuple[int, int]]:
    """Same definition as run_tm_focus_optimization.peak_to_background_ratio (used if the pipeline lacks focus_pbr)."""
    intensity = np.asarray(intensity, dtype=np.float64)
    h, w = intensity.shape
    yy, xx = np.ogrid[:h, :w]
    near = (np.abs(yy - int(target_yx[0])) <= search_radius) & (np.abs(xx - int(target_yx[1])) <= search_radius)
    search = support & near if (support & near).any() else support
    py, px = np.unravel_index(int(np.argmax(np.where(search, intensity, -np.inf))), intensity.shape)
    window = np.zeros((h, w), dtype=bool)
    window[max(0, py - half_width):py + half_width + 1, max(0, px - half_width):px + half_width + 1] = True
    return float(intensity[py, px] / intensity[support & ~window].mean()), (int(py), int(px))


# =====================================================================================
# Registration: the polarizer displaces the image, differently at each angle
# =====================================================================================
def estimate_shift(meas: np.ndarray, ref: np.ndarray, max_shift: float) -> Tuple[float, float, float]:
    """
    Displacement d = (dy, dx) with meas(r + d) ~ ref(r), from the cross-correlation summed over
    the leading axes (SLM patterns). Sub-pixel by a parabolic fit. Returns (dy, dx, peak height).
    """
    meas = np.asarray(meas, dtype=np.float64).reshape(-1, *np.shape(meas)[-2:])
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, *np.shape(ref)[-2:])
    h, w = meas.shape[-2:]
    corr = np.zeros((h, w))
    for m, p in zip(meas, ref):
        m = (m - m.mean()) / (m.std() + 1e-30)
        p = (p - p.mean()) / (p.std() + 1e-30)
        corr += np.fft.ifft2(np.fft.fft2(m) * np.conj(np.fft.fft2(p))).real / (h * w)
    corr /= len(meas)
    dy_idx = np.fft.fftfreq(h, 1.0 / h)
    dx_idx = np.fft.fftfreq(w, 1.0 / w)
    allowed = (np.abs(dy_idx)[:, None] <= max_shift) & (np.abs(dx_idx)[None, :] <= max_shift)
    iy, ix = np.unravel_index(int(np.argmax(np.where(allowed, corr, -np.inf))), corr.shape)

    def _sub(cm: float, c0: float, cp: float) -> float:
        den = cm - 2.0 * c0 + cp
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))

    oy = _sub(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    ox = _sub(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    return float(dy_idx[iy] + oy), float(dx_idx[ix] + ox), float(corr[iy, ix])


def shift_image(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Return out(r) = image(r + d) (bilinear, zero outside): undoes a displacement d."""
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    img = np.ascontiguousarray(image, dtype=np.float32)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def register_stack(raw: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """raw: (n_angles, K, H, W); shifts: (n_angles, 2) -> registered stack."""
    out = np.empty(raw.shape, dtype=np.float64)
    for ia in range(raw.shape[0]):
        for k in range(raw.shape[1]):
            out[ia, k] = shift_image(raw[ia, k], shifts[ia, 0], shifts[ia, 1])
    return out


def shift_at_angle(angle_deg: float, angles_deg: Sequence[float], shifts: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Displacement at a polarizer angle, from the displacements measured at ``angles_deg``.
    Returns (shift, measured): the measured value if that angle was measured; otherwise the
    value of d(a) = c + p cos a + q sin a fitted to the measurements (a wedged polarizer moves
    the image on a circle, once per full turn).
    """
    ang = np.asarray(angles_deg, dtype=np.float64)
    dist = np.abs((ang - angle_deg + 180.0) % 360.0 - 180.0)
    if dist.min() < 1e-6:
        return np.asarray(shifts[int(np.argmin(dist))], dtype=np.float64), True
    if len(ang) >= 3:
        a = np.radians(ang)
        coef = np.linalg.lstsq(np.stack([np.ones_like(a), np.cos(a), np.sin(a)], axis=1), np.asarray(shifts), rcond=None)[0]
        t = np.radians(angle_deg)
        return np.array([1.0, np.cos(t), np.sin(t)]) @ coef, False
    return np.asarray(shifts[int(np.argmin(dist))], dtype=np.float64), False


# =====================================================================================
# Low-order aberration added by the polarizer (pupil phase applied to the model field)
# =====================================================================================
class PupilAberration:
    """
    Phase sum_j c_j Z_j(k) in the Fourier domain of the image-plane field (the NA disk is the
    pupil). Z_j: Noll 4 (defocus), 5, 6 (astigmatism) and, for n_zernike = 6, 7, 8 (coma), 11.

    For the Fourier-plane camera the roles are exchanged: the conjugate domain of the Fourier-plane
    field is the fibre facet, where the field is limited to the core disk. Pass the grid pitch in
    spatial frequency as ``pixel_pitch_m``, the core radius as ``na`` and 1 as ``wavelength_m``
    (see ``make_pupil_aberration``); the phase is then normalized to the core disk.
    """

    def __init__(self, n_grid: int, pixel_pitch_m: float, na: float, wavelength_m: float, n_zernike: int, device):
        f = torch.fft.fftfreq(n_grid, d=pixel_pitch_m)
        fy, fx = torch.meshgrid(f, f, indexing="ij")
        rho = torch.sqrt(fx ** 2 + fy ** 2) / (na / wavelength_m)
        phi = torch.atan2(fy, fx)
        basis = [np.sqrt(3) * (2 * rho ** 2 - 1), np.sqrt(6) * rho ** 2 * torch.sin(2 * phi), np.sqrt(6) * rho ** 2 * torch.cos(2 * phi),
                 np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * torch.sin(phi), np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * torch.cos(phi),
                 np.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1)]
        n_zernike = int(max(0, min(n_zernike, len(basis))))
        inside = (rho <= 1.05).to(torch.float32)
        self.basis = torch.stack([b.to(torch.float32) * inside for b in basis[:n_zernike]]).to(device) if n_zernike else None
        self.pupil_radius_px = float(na / wavelength_m * n_grid * pixel_pitch_m)

    def apply(self, e: torch.Tensor, coeffs: Optional[torch.Tensor]) -> torch.Tensor:
        """e: (..., H, W) complex; coeffs: (n_zernike,) radians or None."""
        if coeffs is None or self.basis is None:
            return e
        phase = torch.tensordot(coeffs.to(self.basis.dtype), self.basis, dims=1)
        return torch.fft.ifft2(torch.fft.fft2(e) * torch.exp(1j * phase))


def _masked_pcc_torch(pred: torch.Tensor, meas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Pearson correlation over the leading axes, pixels restricted to mask."""
    p = pred[..., mask]
    m = meas[..., mask]
    p = p - p.mean(dim=-1, keepdim=True)
    m = m - m.mean(dim=-1, keepdim=True)
    return ((p * m).sum(-1) / (p.norm(dim=-1) * m.norm(dim=-1) + 1e-30)).mean()


def _mix(u: np.ndarray, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lab components (U e)_x, (U e)_y of a field e with polarization on axis 1."""
    u_t = torch.as_tensor(u, dtype=e.dtype, device=e.device)
    return u_t[0, 0] * e[:, 0] + u_t[0, 1] * e[:, 1], u_t[1, 0] * e[:, 0] + u_t[1, 1] * e[:, 1]


def predict_polarizer_images(e_model: torch.Tensor, u: np.ndarray, angles_deg: Sequence[float],
                             aberration: Optional[PupilAberration] = None, coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(K, 2, H, W) model field -> (n_angles, K, H, W) predicted polarizer images (arbitrary units)."""
    e = aberration.apply(e_model, coeffs) if aberration is not None else e_model
    ex, ey = _mix(u, e)
    return torch.stack([(float(np.cos(np.radians(a))) * ex + float(np.sin(np.radians(a))) * ey).abs() ** 2 for a in angles_deg])


def fit_aberration(e_model: torch.Tensor, u: np.ndarray, angles_deg: Sequence[float], registered: np.ndarray,
                   mask: np.ndarray, aberration: PupilAberration, iters: int = 200, lr: float = 0.03) -> torch.Tensor:
    """Pupil-phase coefficients that maximize the correlation between predicted and measured polarizer images."""
    meas = torch.as_tensor(registered, dtype=torch.float32, device=e_model.device)
    mask_t = torch.as_tensor(mask, device=e_model.device)
    coeffs = nn.Parameter(torch.zeros(aberration.basis.shape[0], dtype=torch.float32, device=e_model.device))
    opt = optim.Adam([coeffs], lr=lr)
    best, best_val = coeffs.detach().clone(), -np.inf
    for _ in range(iters):
        opt.zero_grad()
        val = _masked_pcc_torch(predict_polarizer_images(e_model, u, angles_deg, aberration, coeffs), meas, mask_t)
        if val.item() > best_val:
            best_val, best = val.item(), coeffs.detach().clone()
        (-val).backward()
        opt.step()
    return best


# =====================================================================================
# Frame estimation with registration (and optional aberration fit)
# =====================================================================================
def solve_frame(raw: np.ndarray, angles_deg: Sequence[float], e_model: torch.Tensor, support: np.ndarray,
                cfg: PolarizationConfig, aberration: Optional[PupilAberration]) -> Dict:
    """
    raw: (n_angles, K, H, W) camera frames (model orientation, same exposure).
    e_model: (K, 2, H, W) complex torch tensor, model field of the same SLM patterns.
    Returns U, the per-angle displacements, the aberration coefficients and quality figures.
    """
    n_angles = raw.shape[0]
    h, w = raw.shape[-2:]
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(support.sum() / np.pi)
    inner = (yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2 <= max(radius - cfg.max_shift_px / 2.0 - 2.0, 4.0) ** 2
    mask = support & inner                                   # stay clear of the edge that the displacement moves
    mask_t = torch.as_tensor(mask, device=e_model.device)

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _quality(u, shifts, coeffs) -> List[float]:
        pred = predict_polarizer_images(e_model, u, angles_deg, aberration, coeffs)
        meas_t = torch.as_tensor(register_stack(raw, shifts), dtype=torch.float32, device=e_model.device)
        return [float(_masked_pcc_torch(pred[i], meas_t[i], mask_t)) for i in range(n_angles)]

    def _refine(u, shifts, coeffs, rounds: int):
        est = None
        for _ in range(rounds):
            pred = _np(predict_polarizer_images(e_model, u, angles_deg, aberration, coeffs))
            shifts = np.array([estimate_shift(raw[i], pred[i], cfg.max_shift_px)[:2] for i in range(n_angles)])
            e_used = _np(aberration.apply(e_model, coeffs)) if (aberration is not None and coeffs is not None) else _np(e_model)
            est = estimate_frame(register_stack(raw, shifts), angles_deg, e_used, mask)
            u = est["U"]
        return est, shifts

    # 1) coarse displacement against the polarization-summed model intensity
    s0 = (np.abs(_np(e_model)) ** 2).sum(axis=1)                                               # (K, H, W)
    shifts = np.array([estimate_shift(raw[i], s0, cfg.max_shift_px)[:2] for i in range(n_angles)])
    est = estimate_frame(register_stack(raw, shifts), angles_deg, _np(e_model), mask)
    # 2) refine the displacement against the predicted polarizer images, re-solve
    est, shifts = _refine(est["U"], shifts, None, rounds=2)
    pcc_plain = _quality(est["U"], shifts, None)
    result = {"estimate": est, "shifts_px": shifts, "aberration_coeffs_rad": None, "pcc_per_angle": pcc_plain,
              "pcc_per_angle_without_aberration_fit": pcc_plain, "aberration_fit_used": False, "aberration_fit_candidate": None}

    # 3) optional low-order aberration of the polarizer; kept only if it clearly helps
    if cfg.fit_aberrations and aberration is not None and aberration.basis is not None:
        coeffs, est_ab, shifts_ab = None, est, shifts
        for _ in range(2):
            coeffs = fit_aberration(e_model, est_ab["U"], angles_deg, register_stack(raw, shifts_ab), mask, aberration)
            est_ab, shifts_ab = _refine(est_ab["U"], shifts_ab, coeffs, rounds=1)
        pcc_ab = _quality(est_ab["U"], shifts_ab, coeffs)
        result["aberration_fit_candidate"] = {"coeffs_rad": [float(c) for c in coeffs], "pcc_per_angle": pcc_ab}
        if float(np.mean(pcc_ab)) - float(np.mean(pcc_plain)) >= cfg.aberration_min_gain:
            result.update({"estimate": est_ab, "shifts_px": shifts_ab, "aberration_coeffs_rad": coeffs,
                           "pcc_per_angle": pcc_ab, "aberration_fit_used": True})
    return result


# =====================================================================================
# Two foci in the two lab polarizations
# =====================================================================================
def _window_mask(n: int, yx: Tuple[int, int], half: int, device) -> torch.Tensor:
    m = torch.zeros((n, n), dtype=torch.bool, device=device)
    m[max(0, yx[0] - half):yx[0] + half + 1, max(0, yx[1] - half):yx[1] + half + 1] = True
    return m


def model_plane_field(tm_system, slm_phase: torch.Tensor, plane: str) -> torch.Tensor:
    """Model field at the camera of ``plane``, (B, 2, H, W), polarization (x', y') on axis 1."""
    e = tm_system.forward_model_slm(slm_phase, return_complex_field=True, active_polarization="both")[:, PLANES[plane]["index"]]
    if plane == "fourier" and not getattr(tm_system, "enforce_symmetric_tm", False):
        # forward_model_slm returns the Fourier field AFTER the model's Fourier-domain Zernike phase,
        # which belongs to the path to the image camera; the Fourier camera sees the field before it
        # (I_fourier in TM_retrieve). The phase is common to x' and y', so intensities and Stokes
        # parameters are the same either way; it matters only for the optional aberration fit.
        e = tm_system.zernike_layer(e, -tm_system.zernike_fourier_domain.detach())
    return e


def make_pupil_aberration(tm_system, plane: str, n: int, n_zernike: int, support_radius_px: float) -> PupilAberration:
    """Low-order aberration basis for the camera plane (see PupilAberration)."""
    fp = tm_system.fiber_params
    na = float(np.sqrt(fp["n1"] ** 2 - fp["n0"] ** 2))
    wavelength = float(fp["wavelength"])
    if plane == "image":
        return PupilAberration(n_grid=n, pixel_pitch_m=float(tm_system.grid_params["x_range"][1]) / (n / 2.0),
                               na=na, wavelength_m=wavelength, n_zernike=n_zernike, device=tm_system.device)
    # Fourier plane: the NA disk (radius NA / wavelength in spatial frequency) spans support_radius_px pixels.
    freq_pitch = (na / wavelength) / float(support_radius_px)
    return PupilAberration(n_grid=n, pixel_pitch_m=freq_pitch, na=float(fp["a"]), wavelength_m=1.0,
                           n_zernike=n_zernike, device=tm_system.device)


def lab_field(tm_system, slm_phase: torch.Tensor, u: np.ndarray, aberration: Optional[PupilAberration],
              coeffs: Optional[torch.Tensor], plane: str = "image") -> torch.Tensor:
    """Camera-plane field in the lab frame, (B, 2, H, W): U applied to the (aberrated) model field."""
    e = model_plane_field(tm_system, slm_phase, plane)
    if aberration is not None:
        e = aberration.apply(e, coeffs)
    return torch.stack(_mix(u, e), dim=1)


def optimize_two_foci(tm_system, u: np.ndarray, target_x_yx: Tuple[int, int], target_y_yx: Tuple[int, int],
                      support: np.ndarray, cfg: PolarizationConfig, aberration: Optional[PupilAberration] = None,
                      coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Phase pattern (n, n) with focus A at target_x_yx polarized along lab x and focus B at
    target_y_yx polarized along lab y. Objective: F_x(A) * F_y(B), F_c = window mean / background
    mean of |E_c|^2; with cross_pol_penalty each factor is divided by (1 + cross-polarized
    window mean / background mean), which keeps each focus polarization-pure.
    """
    for p in tm_system.parameters():
        p.requires_grad = False
    tm_system.eval()
    device = tm_system.device
    n = int(tm_system.grid_params["ny"])
    sup = torch.as_tensor(support, device=device)
    ma, mb = _window_mask(n, target_x_yx, cfg.window_half_width, device), _window_mask(n, target_y_yx, cfg.window_half_width, device)
    bg_a, bg_b = sup & ~ma, sup & ~mb
    gen = torch.Generator().manual_seed(int(cfg.random_seed) + 17)
    phase = nn.Parameter((2 * torch.pi * torch.rand((1, n, n), generator=gen)).to(device))
    opt = optim.Adam([phase], lr=cfg.slm_opt_lr)
    for i in range(int(cfg.slm_opt_iters)):
        opt.zero_grad()
        e = lab_field(tm_system, phase, u, aberration, coeffs, cfg.plane)[0]
        ix, iy = e[0].abs() ** 2, e[1].abs() ** 2
        fx, fy = ix[ma].mean() / ix[bg_a].mean(), iy[mb].mean() / iy[bg_b].mean()
        if cfg.cross_pol_penalty:
            fx = fx / (1.0 + iy[ma].mean() / iy[bg_b].mean())
            fy = fy / (1.0 + ix[mb].mean() / ix[bg_a].mean())
        (-(fx * fy)).backward()
        opt.step()
        if (i + 1) % 50 == 0:
            print(f"  [two foci] iteration {i + 1}/{cfg.slm_opt_iters}: objective factors {fx.item():.1f} (A), {fy.item():.1f} (B)")
    return torch.remainder(phase.detach(), 2 * torch.pi)[0]


def predicted_two_foci(e_lab: torch.Tensor, target_a: Tuple[int, int], target_b: Tuple[int, int], half: int) -> Dict:
    """Polarization purity of the two foci predicted by the model; e_lab: (2, H, W) lab-frame field."""
    n = e_lab.shape[-1]
    ix, iy = e_lab[0].abs() ** 2, e_lab[1].abs() ** 2
    ma, mb = _window_mask(n, target_a, half, e_lab.device), _window_mask(n, target_b, half, e_lab.device)
    tau_a = float(ix[ma].sum() / (ix[ma].sum() + iy[ma].sum()))
    tau_b = float(ix[mb].sum() / (ix[mb].sum() + iy[mb].sum()))
    return {"transmitted_fraction_A_at_x": tau_a, "transmitted_fraction_B_at_x": tau_b, "separation": tau_a - tau_b}


def focus_metrics(registered: Dict[float, np.ndarray], target_a: Tuple[int, int], target_b: Tuple[int, int],
                  support: np.ndarray, pbr_fn: Optional[Callable] = None, search_radius: int = 3) -> Dict:
    """
    registered: {relative polarizer angle (deg): registered frame}.
    Each focus is located once (brightest pixel near its target in the sum over angles); its
    power is the dark-subtracted sum over the 3x3 window there, at every angle.
    With 0 and 90 deg present: transmitted fractions tau = P_0 / (P_0 + P_90) and the
    separation tau_A - tau_B (1 = A only at x, B only at y).
    With >= 3 angles: Malus fit per focus -> azimuth and degree of linear polarization.
    """
    angles = sorted(registered)
    stack = np.stack([np.asarray(registered[a], dtype=np.float64) for a in angles])
    dark = np.array([np.median(f[~support]) if (~support).any() else 0.0 for f in stack])
    total = (stack - dark[:, None, None]).sum(axis=0)
    h, w = support.shape
    yy, xx = np.mgrid[:h, :w]
    pbr_fn = pbr_fn or (lambda img, yx: peak_to_background_ratio(img, support, yx)[0])
    out: Dict = {"angles_deg": angles, "dark_level": dark.tolist()}
    powers = {}
    for name, tgt in (("A", target_a), ("B", target_b)):
        near = (np.abs(yy - tgt[0]) <= search_radius) & (np.abs(xx - tgt[1]) <= search_radius)
        py, px = np.unravel_index(int(np.argmax(np.where(near, total, -np.inf))), total.shape)
        win = (np.abs(yy - py) <= 1) & (np.abs(xx - px) <= 1)
        powers[name] = np.array([(stack[i][win] - dark[i]).sum() for i in range(len(angles))])
        out[name] = {"peak_yx": [int(py), int(px)], "window_power": powers[name].tolist(),
                     "pbr_behind_polarizer": [float(pbr_fn(stack[i], (int(py), int(px)))) for i in range(len(angles))]}
        if len(angles) >= 3:
            a = np.radians(np.array(angles))
            design = 0.5 * np.stack([np.ones_like(a), np.cos(2 * a), np.sin(2 * a)], axis=1)
            if np.linalg.matrix_rank(design) == 3:
                c = np.linalg.lstsq(design, powers[name], rcond=None)[0]
                out[name]["azimuth_deg"] = float(np.degrees(0.5 * np.arctan2(c[2], c[1])) % 180.0)
                out[name]["degree_of_linear_polarization"] = float(np.hypot(c[1], c[2]) / c[0])
    if 0.0 in registered and 90.0 in registered:
        i0, i90 = angles.index(0.0), angles.index(90.0)
        tau_a = powers["A"][i0] / (powers["A"][i0] + powers["A"][i90])
        tau_b = powers["B"][i0] / (powers["B"][i0] + powers["B"][i90])
        out.update({"transmitted_fraction_A_at_x": float(tau_a), "transmitted_fraction_B_at_x": float(tau_b),
                    "separation": float(tau_a - tau_b),
                    "leak_B_over_A_at_x": float(powers["B"][i0] / powers["A"][i0]),
                    "leak_A_over_B_at_y": float(powers["A"][i90] / powers["B"][i90])})
    return out


# =====================================================================================
# Lab procedure (hardware through TMFocusPipeline)
# =====================================================================================
def _saturation_level(pipeline) -> int:
    return 4095 if str(getattr(pipeline.cfg, "camera_pixel_format", "Mono12")) == "Mono12" else 255


def _capture_plane(pipeline, exposure: float, plane: str) -> Tuple[np.ndarray, bool]:
    """One frame of the camera of ``plane`` in model orientation (crop, resize, flip for the image plane) and whether the raw frame saturated."""
    spec = PLANES[plane]
    raw = np.squeeze(np.array(pipeline.capture_hdr_images(pipeline.cameras[spec["camera"]], [exposure])))
    saturated = bool(raw.max() >= _saturation_level(pipeline))
    proc = pipeline.crop_and_resize_image(raw, spec["crop"]).astype(np.float32)
    if spec["flip"]:
        proc = np.flip(proc)
    return np.ascontiguousarray(proc), saturated


def _speckle_masks(pipeline, n_patterns: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """uint8 (K, n, n) SLM patterns in the model convention: probe masks of the TM run if available."""
    try:
        if getattr(pipeline.cfg, "use_optimized_patterns", False):
            masks = pipeline._load_optimized_small_masks()
            idx = np.sort(rng.choice(masks.shape[0], size=n_patterns, replace=False))
            print(f"Speckle patterns: probe masks {[int(i) for i in idx]} of the TM acquisition")
            return np.ascontiguousarray(masks[idx], dtype=np.uint8)
    except Exception as exc:                                   # missing file etc.: fall back to random patterns
        print(f"Could not load the probe masks ({exc}); using random patterns instead.")
    coarse = rng.integers(0, 256, size=(n_patterns, n // 4, n // 4), dtype=np.uint8)
    return np.kron(coarse, np.ones((4, 4), dtype=np.uint8))


def _phase_from_uint8(tm_system, phase_uint8: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(phase_uint8.astype(np.float32) * (2 * np.pi / 256.0), device=tm_system.device)


def _support_mask(pipeline, n: int, plane: str = "image") -> np.ndarray:
    """Fibre core disk (image plane) or NA disk (Fourier plane) on the model grid."""
    if hasattr(pipeline, "support_mask"):
        return np.asarray(pipeline.support_mask(plane, (n, n)), dtype=bool)
    yy, xx = np.ogrid[:n, :n]
    return (yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2 <= (n / 2.2) ** 2


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"real": obj.real.tolist(), "imag": obj.imag.tolist()}
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _save_png16_common(frames: Dict[float, np.ndarray], out_dir: Path, stem: str) -> None:
    """16-bit PNGs with ONE scale for all angles, so that the x and y images can be compared."""
    peak = max(float(np.max(f)) for f in frames.values())
    for ang, f in frames.items():
        u16 = np.clip(np.round(np.asarray(f, dtype=np.float64) * (65535.0 / max(peak, 1e-12))), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_dir / f"{stem}_pol_{ang:g}deg.png"), u16)


def _save_overview(path: Path, predicted: Dict[float, np.ndarray], measured: Dict[float, np.ndarray],
                   target_a: Tuple[int, int], target_b: Tuple[int, int]) -> None:
    """Quick-look figure: predicted (top) and measured, registered (bottom) image per polarizer angle."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        angles = sorted(measured)
        fig, axes = plt.subplots(2, len(angles), figsize=(3.2 * len(angles), 6.4), squeeze=False)
        for row, (label, data) in enumerate((("predicted", predicted), ("measured", measured))):
            vmax = max(float(np.max(data[a])) for a in angles)
            for col, ang in enumerate(angles):
                ax = axes[row, col]
                ax.imshow(data[ang], cmap="inferno", vmin=0, vmax=vmax)
                for tgt, colour in ((target_a, "cyan"), (target_b, "lime")):
                    ax.plot(tgt[1], tgt[0], marker="o", ms=14, mfc="none", mec=colour, mew=0.8)
                ax.set_title(f"{label}, polarizer {ang:g} deg", fontsize=9)
                ax.set_axis_off()
        fig.suptitle("cyan: focus A (lab x)   green: focus B (lab y)", fontsize=9)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
    except Exception as exc:                                   # a plotting problem must not cost the measurement
        print(f"  (overview figure not saved: {exc})")


def run_polarization_axes(pipeline, tm_system, stage, cfg: PolarizationConfig, min_exposure: float = 0.0) -> Dict:
    """Steps 2-5 of the module docstring. ``pipeline.setup()`` and ``stage.connect()`` must already have run."""
    out_dir = Path(pipeline.cfg.output_dir) / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(tm_system.grid_params["ny"])
    plane = cfg.plane
    support = _support_mask(pipeline, n, plane)
    rng = np.random.default_rng(cfg.random_seed)
    aberration = make_pupil_aberration(tm_system, plane, n, cfg.n_zernike if cfg.fit_aberrations else 0,
                                       support_radius_px=float(np.sqrt(support.sum() / np.pi)))
    camera = pipeline.cameras[PLANES[plane]["camera"]]
    summary: Dict = {"config": asdict(cfg), "plane": plane, "camera_serial": str(camera.GetDeviceInfo().GetSerialNumber()),
                     "pupil_radius_fft_px": aberration.pupil_radius_px}
    print(f"Camera behind the polarizer: {plane} plane, serial {summary['camera_serial']}")
    pbr_fn = (lambda img, yx: pipeline.focus_pbr(img, plane=plane, target_yx=yx)[0]) if hasattr(pipeline, "focus_pbr") else None

    # ---------------- 2. polarizer images of speckle patterns
    masks = _speckle_masks(pipeline, cfg.n_speckle_patterns, n, rng)
    with torch.no_grad():
        e_model = model_plane_field(tm_system, _phase_from_uint8(tm_system, masks), plane)   # (K, 2, n, n)
    exposure = float(cfg.speckle_exposure) or (float(getattr(pipeline.cfg, PLANES[plane]["exposure_field"])[0])
                                               * float(pipeline.cfg.gain))
    angles = [float(a) for a in cfg.frame_angles_deg]
    for _attempt in range(5):
        raw = np.zeros((len(angles), len(masks), n, n), dtype=np.float64)
        any_saturated = False
        for ia, ang in enumerate(angles):
            stage.move_to(cfg.polarizer_x_angle_deg + ang)
            for k in range(len(masks)):
                pipeline.project_phase_mask_to_slm(pipeline.pack_phase_to_full_slm(masks[k]))
                raw[ia, k], sat = _capture_plane(pipeline, exposure, plane)
                any_saturated |= sat
            print(f"Polarizer at {ang:g} deg: {len(masks)} speckle frames (exposure {exposure:g}, mean level {raw[ia].mean():.1f})")
        if not any_saturated:
            break
        exposure = max(exposure / 2.0, min_exposure)
        print(f"Saturated speckle frame: repeating all angles at exposure {exposure:g}")
    else:
        raise RuntimeError("Speckle frames stay saturated; set a lower speckle_exposure.")
    np.savez_compressed(out_dir / "polarizer_speckle_frames.npz", frames=raw.astype(np.float32), angles_deg=np.array(angles),
                        slm_masks_uint8=masks, exposure=exposure, polarizer_x_angle_deg=cfg.polarizer_x_angle_deg)

    # ---------------- 3. displacement, aberration, Jones matrix
    sol = solve_frame(raw, angles, e_model, support, cfg, aberration)
    est, coeffs, frame_shifts = sol["estimate"], sol["aberration_coeffs_rad"], sol["shifts_px"]
    u = est["U"]
    print("\nPolarization frame of the retrieved TM (E_lab = c U E_model):")
    print("  image displacement per polarizer angle (dy, dx) px: "
          + ", ".join(f"{a:g} deg: ({s[0]:+.2f}, {s[1]:+.2f})" for a, s in zip(angles, frame_shifts)))
    print(f"  predicted vs measured polarizer images, PCC per angle: {[round(v, 3) for v in sol['pcc_per_angle']]}")
    if sol["aberration_fit_candidate"]:
        cand = sol["aberration_fit_candidate"]
        print(f"  aberration fit {'USED' if sol['aberration_fit_used'] else 'not used'}: coefficients (rad) "
              f"{[round(c, 3) for c in cand['coeffs_rad']]}, mean PCC "
              f"{np.mean(sol['pcc_per_angle_without_aberration_fit']):.3f} -> {np.mean(cand['pcc_per_angle']):.3f}")
    print(f"  explained fraction of the polarizer signal: {est['explained_fraction']:.3f}")
    print(f"  x' axis of the TM seen in the lab: azimuth {est['xprime_azimuth_deg']:.1f} deg, ellipticity {est['xprime_ellipticity_deg']:+.1f} deg")
    print(f"  if only the polarizer were rotated (no correction): best separation {est['separation_without_correction']:.3f} "
          f"at {est['xprime_azimuth_deg']:.1f} deg; rotation-only angle {est['rotation_only_angle_deg']:.1f} deg, "
          f"out-of-plane part {est['out_of_plane']:.3f}")
    if np.mean(sol["pcc_per_angle"]) < 0.5:
        print("  WARNING: the model explains the polarizer images poorly. Check exposure, focus, and drift since the TM run.")
    if np.abs(frame_shifts).max() > 0.6 * cfg.max_shift_px:
        print("  WARNING: large image displacement; re-centre the polarizer or the camera crop, or raise max_shift_px.")
    np.save(out_dir / "jones_matrix_U.npy", u)
    summary["frame"] = {**est, "frame_angles_deg": angles, "shifts_px": frame_shifts, "pcc_per_angle": sol["pcc_per_angle"],
                        "pcc_per_angle_without_aberration_fit": sol["pcc_per_angle_without_aberration_fit"],
                        "aberration_fit_used": sol["aberration_fit_used"], "aberration_coeffs_rad": coeffs,
                        "aberration_fit_candidate": sol["aberration_fit_candidate"], "speckle_exposure": exposure}

    # ---------------- 4-5. two foci along lab x / lab y, recorded behind the polarizer
    target_a = tuple(int(v) for v in cfg.target_x_pol_yx) if cfg.target_x_pol_yx else (n // 2 - n // 8, n // 2 - n // 16)
    target_b = tuple(int(v) for v in cfg.target_y_pol_yx) if cfg.target_y_pol_yx else (n // 2 + n // 8, n // 2 + n // 16)
    design_coeffs = coeffs if cfg.compensate_aberration_in_design else None
    designs = ([("tm_axes_uncorrected", np.eye(2, dtype=complex))] if cfg.also_uncorrected else []) + [("lab_axes", u)]
    verify = [float(a) for a in cfg.verify_angles_deg]
    summary["two_foci"] = {"target_A_x_pol_yx": list(target_a), "target_B_y_pol_yx": list(target_b)}
    for tag, u_design in designs:                             # the corrected pattern is done last and stays on the SLM
        print(f"\nTwo foci ({tag}): A at {target_a} along x, B at {target_b} along y")
        phase_np = optimize_two_foci(tm_system, u_design, target_a, target_b, support, cfg, aberration, design_coeffs).cpu().numpy()
        phase_u8 = pipeline.phase_to_uint8(phase_np)
        full_mask = pipeline.pack_phase_to_full_slm(phase_u8)
        np.save(out_dir / f"two_foci_{tag}_phase_rad.npy", phase_np)
        np.save(out_dir / f"two_foci_{tag}_phase_uint8.npy", phase_u8)
        np.save(out_dir / f"two_foci_{tag}_full_slm_mask_uint8.npy", full_mask)
        if hasattr(pipeline, "save_phase_hsv_png"):
            pipeline.save_phase_hsv_png(phase_np, out_dir / f"two_foci_{tag}_phase_hsv.png")

        # model prediction for the pattern as displayed (8 bit), seen through the polarizer
        with torch.no_grad():
            e_lab = lab_field(tm_system, _phase_from_uint8(tm_system, phase_u8).unsqueeze(0), u, aberration, coeffs, plane)
            predicted = predicted_two_foci(e_lab[0], target_a, target_b, cfg.window_half_width)
            pred_imgs = predict_polarizer_images(e_lab, np.eye(2), verify).cpu().numpy()[:, 0]

        pipeline.project_phase_mask_to_slm(full_mask)
        foci_exposure = max(float(cfg.foci_exposure), min_exposure)
        while True:
            frames, any_saturated = {}, False
            for ang in verify:
                stage.move_to(cfg.polarizer_x_angle_deg + ang)
                frames[ang], sat = _capture_plane(pipeline, foci_exposure, plane)
                any_saturated |= sat
            if not any_saturated or foci_exposure <= max(min_exposure, 1e-3):
                break
            foci_exposure = max(foci_exposure / 2.0, min_exposure)
            print(f"  saturated focus frame: repeating at exposure {foci_exposure:g}")
        if any_saturated:
            print("  WARNING: focus frames saturate at the minimum exposure; the metrics are biased.")

        # undo the polarizer displacement with the values measured on the speckle frames
        registered, shifts, shifts_check = {}, {}, {}
        for i, ang in enumerate(verify):
            d, measured_here = shift_at_angle(ang, angles, frame_shifts)
            if not measured_here:
                print(f"  note: displacement at {ang:g} deg interpolated from the speckle angles")
            shifts[ang] = [float(d[0]), float(d[1])]
            shifts_check[ang] = [float(v) for v in estimate_shift(frames[ang], pred_imgs[i], cfg.max_shift_px)[:2]]
            registered[ang] = shift_image(frames[ang], d[0], d[1])
        metrics = focus_metrics(registered, target_a, target_b, support, pbr_fn)
        np.savez_compressed(out_dir / f"two_foci_{tag}_frames.npz", angles_deg=np.array(verify),
                            frames=np.stack([frames[a] for a in verify]), registered=np.stack([registered[a] for a in verify]),
                            predicted=pred_imgs, exposure=foci_exposure)
        _save_png16_common(frames, out_dir, f"im_two_foci_{tag}")
        _save_overview(out_dir / f"two_foci_{tag}_overview.png", dict(zip(verify, pred_imgs)), registered, target_a, target_b)
        print(f"  predicted: A passes {100 * predicted['transmitted_fraction_A_at_x']:.1f}% at x, B passes "
              f"{100 * (1 - predicted['transmitted_fraction_B_at_x']):.1f}% at y; separation {predicted['separation']:+.3f}")
        if "separation" in metrics:
            print(f"  measured:  A passes {100 * metrics['transmitted_fraction_A_at_x']:.1f}% at x, B passes "
                  f"{100 * (1 - metrics['transmitted_fraction_B_at_x']):.1f}% at y; separation {metrics['separation']:+.3f}; "
                  f"leak B/A at x {metrics['leak_B_over_A_at_x']:.3f}, A/B at y {metrics['leak_A_over_B_at_y']:.3f}")
        summary["two_foci"][tag] = {"predicted": predicted, "measured": metrics, "exposure": foci_exposure, "saturated": any_saturated,
                                    "frame_shifts_px": {f"{a:g}": s for a, s in shifts.items()},
                                    "frame_shifts_from_foci_px": {f"{a:g}": s for a, s in shifts_check.items()}}

    stage.move_to(cfg.polarizer_x_angle_deg)
    with open(out_dir / "polarization_axes_summary.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
    print(f"\nSaved everything to {out_dir}")
    return summary


def test_stage(stage, cfg: PolarizationConfig) -> None:
    """Moves the stage through the angles of a run and prints the read-back position."""
    for ang in list(cfg.frame_angles_deg) + [0.0]:
        target = cfg.polarizer_x_angle_deg + float(ang)
        t0 = time.time()
        stage.move_to(target)
        print(f"requested {target:8.3f} deg -> stage reports {stage.position():8.3f} deg ({time.time() - t0:.2f} s)")


# =====================================================================================
# Self-test of the math (no hardware, no TM)
# =====================================================================================
def self_test() -> None:
    rng = np.random.default_rng(1)
    n, k = 64, 3
    pitch, na, lam = 0.43e-6, 0.22, 632.8e-9
    ab = PupilAberration(n, pitch, na, lam, 3, "cpu")
    f = np.fft.fftfreq(n, d=pitch)
    pupil = (np.hypot(*np.meshgrid(f, f, indexing="ij")) <= na / lam)
    yy, xx = np.ogrid[:n, :n]
    core = (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= (n / 2.2) ** 2

    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    u_true = q / np.sqrt(np.linalg.det(q))
    r_true = mueller_rotation(u_true)
    e_model = np.fft.ifft2((rng.normal(size=(k, 2, n, n)) + 1j * rng.normal(size=(k, 2, n, n))) * pupil) * core
    angles = [0.0, 45.0, 90.0, 135.0]
    coeff_true = torch.tensor([0.6, -0.3, 0.2])
    e_cam = ab.apply(torch.as_tensor(e_model, dtype=torch.complex64), coeff_true).numpy()      # polarizer adds an aberration
    e_cam = 2.0 * apply_jones(u_true, e_cam)
    true_shifts = np.array([[3.0 * np.cos(np.radians(a + 20)), 3.0 * np.sin(np.radians(a + 20))] for a in angles])
    raw = np.zeros((len(angles), k, n, n))
    for i, a in enumerate(angles):
        for j in range(k):
            clean = polarizer_intensity(e_cam[j], a)
            raw[i, j] = (shift_image(clean, -true_shifts[i, 0], -true_shifts[i, 1])
                         + 0.03 * clean.mean() * rng.normal(size=(n, n)) + 0.5 * clean.mean())   # noise + camera offset
    cfg = PolarizationConfig(max_shift_px=8.0)
    sol = solve_frame(raw, angles, torch.as_tensor(e_model, dtype=torch.complex64), core, cfg, ab)
    err = np.degrees(np.arccos(np.clip((np.trace(sol["estimate"]["R"] @ r_true.T) - 1) / 2, -1, 1))) / 2
    print(f"frame error {err:.3f} deg | displacement error {np.abs(sol['shifts_px'] - true_shifts).max():.3f} px | "
          f"aberration fit used: {sol['aberration_fit_used']}, coefficients {[round(float(c), 3) for c in sol['aberration_coeffs_rad']]} "
          f"(true {[round(float(c), 3) for c in coeff_true]}) | PCC {np.mean(sol['pcc_per_angle_without_aberration_fit']):.3f} -> {np.mean(sol['pcc_per_angle']):.3f}")
    assert err < 0.5 and np.abs(sol["shifts_px"] - true_shifts).max() < 0.3 and sol["aberration_fit_used"]
    d, measured_here = shift_at_angle(200.0, angles, true_shifts)
    assert not measured_here and np.abs(d - 3.0 * np.array([np.cos(np.radians(220.0)), np.sin(np.radians(220.0))])).max() < 1e-9
    # without aberration and displacement the answer must be exact, for three angles too
    e_cam0 = apply_jones(u_true, e_model)
    for ang_set in ([0.0, 45.0, 90.0, 135.0], [0.0, 60.0, 120.0]):
        imgs = np.stack([np.stack([polarizer_intensity(e_cam0[j], a) for j in range(k)]) for a in ang_set])
        est = estimate_frame(imgs, ang_set, e_model, core)
        e0 = np.degrees(np.arccos(np.clip((np.trace(est["R"] @ r_true.T) - 1) / 2, -1, 1))) / 2
        assert e0 < 1e-3 and est["explained_fraction"] > 0.999999, (ang_set, e0)
    print("self-test passed")


# =====================================================================================
# Entry point
# =====================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Polarization frame of the retrieved TM with a rotating polarizer; two foci in lab x / y.")
    ap.add_argument("--self-test", action="store_true", help="Check the math on synthetic data; no hardware needed.")
    ap.add_argument("--test-stage", action="store_true", help="Only move the stage through the angles and print its position.")
    ap.add_argument("--manual-stage", action="store_true", help="Rotate the polarizer by hand when prompted.")
    ap.add_argument("--plane", type=str, default="fourier", choices=sorted(PLANES),
                    help="Transmission camera behind the polarizer (default: fourier).")
    ap.add_argument("--checkpoint", type=str, default="", help="TM checkpoint (default: PipelineConfig.tm_checkpoint_path).")
    ap.add_argument("--x-angle", type=float, default=0.0, help="Stage angle that defines lab x (deg).")
    ap.add_argument("--frame-angles", type=str, default="0,45,90,135", help="Polarizer angles for the frame, relative to lab x.")
    ap.add_argument("--verify-angles", type=str, default="0,90", help="Polarizer angles at which the two foci are recorded.")
    ap.add_argument("--n-patterns", type=int, default=4, help="Number of speckle SLM patterns used to find the frame.")
    ap.add_argument("--speckle-exposure", type=float, default=0.0, help="Exposure of the speckle frames (0: as in the TM acquisition).")
    ap.add_argument("--foci-exposure", type=float, default=24.0, help="Start exposure of the focus frames (halved on saturation).")
    ap.add_argument("--target-x", type=str, default="", help="y,x of the x-polarized focus (default near the centre).")
    ap.add_argument("--target-y", type=str, default="", help="y,x of the y-polarized focus.")
    ap.add_argument("--no-aberration-fit", action="store_true", help="Do not fit an aberration of the polarizer.")
    ap.add_argument("--also-uncorrected", action="store_true", help="Also record foci designed in the TM's own axes, for comparison.")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    cfg = PolarizationConfig(
        plane=args.plane,
        polarizer_x_angle_deg=args.x_angle,
        frame_angles_deg=tuple(float(v) for v in args.frame_angles.split(",")),
        verify_angles_deg=tuple(float(v) for v in args.verify_angles.split(",")),
        n_speckle_patterns=args.n_patterns, speckle_exposure=args.speckle_exposure, foci_exposure=args.foci_exposure,
        fit_aberrations=not args.no_aberration_fit, also_uncorrected=args.also_uncorrected, checkpoint_path=args.checkpoint,
        target_x_pol_yx=tuple(int(v) for v in args.target_x.split(",")) if args.target_x else None,
        target_y_pol_yx=tuple(int(v) for v in args.target_y.split(",")) if args.target_y else None,
    )
    stage = ManualPolarizerStage() if args.manual_stage else RotatingPolarizerStage()
    if args.test_stage:
        stage.connect()
        try:
            test_stage(stage, cfg)
        finally:
            stage.close()
        return

    # Hardware imports only here, so that --self-test and --test-stage run without cameras or SLM.
    from run_tm_focus_optimization import PipelineConfig, TMFocusPipeline

    pipeline = TMFocusPipeline(PipelineConfig())
    checkpoint = Path(cfg.checkpoint_path or pipeline.cfg.tm_checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"TM checkpoint not found: {checkpoint}")
    device = str(pipeline.cfg.device) if torch.cuda.is_available() else "cpu"
    print(f"Loading TM checkpoint {checkpoint} on {device}")
    tm_system = torch.load(checkpoint, map_location=device, weights_only=False)
    tm_system.device = device

    stage.connect()                      # before the cameras and SLM, so a missing stage driver fails early
    try:
        pipeline.setup()
        try:
            try:
                from create_refs import get_camera_exposure_minimum
                min_exposure = float(get_camera_exposure_minimum(pipeline.cameras[PLANES[cfg.plane]["camera"]]))
            except Exception:
                min_exposure = 0.0
            run_polarization_axes(pipeline, tm_system, stage, cfg, min_exposure=min_exposure)
        finally:
            pipeline.teardown()
    finally:
        stage.close()


if __name__ == "__main__":
    main()
