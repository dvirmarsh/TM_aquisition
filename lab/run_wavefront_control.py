import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import POINTER, c_ubyte, c_uint
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import pypylon.pylon as py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))            # this repository's core modules
sys.path.insert(0, str(REPO_ROOT.parent))     # the private lab repo providing create_refs

from create_refs import create_SLM, create_SLMpy, create_cmeras
from TM_retrieve import TMMeasurementSystem
from metrics_losses import FIBER_CROP_MARGIN, PearsonCorrelationLoss, disk_support_mask, enhancement_factor
from utils import load_real_mmf_data

# Recorded in every JSON summary so new results can be told apart from runs that
# used the old full-frame background.
ENHANCEMENT_DEFINITION = (
    "eta = mean(I over target window) / mean(I over fiber support minus target window); "
    "support = fiber core disk (image plane) or NA disk (Fourier plane)"
)


def _resolve_tm_lib_path(path: str) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(REPO_ROOT / p)


def _resolve_output_h5_path(path: str) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(REPO_ROOT / p)


def output_modes_to_normalized_powers(
    output_mode_dist: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    Guided-mode powers P_k = |c_k|^2 from the TM output coefficients, then
    p_k = P_k / (sum_j P_j + epsilon).

    Args:
        output_mode_dist: From TMMeasurementSystem.slm_to_output_modes,
            shape (N_modes, batch).
        epsilon: Small positive constant for numerical stability.

    Returns:
        p with shape (N_modes, batch), rows sum to ~1 across modes per batch element.
    """
    if output_mode_dist.dim() == 2:
        P = output_mode_dist.abs() ** 2
    else:
        raise ValueError(f"Unexpected output_mode_dist shape {tuple(output_mode_dist.shape)}")
    denom = P.sum(dim=0, keepdim=True) + epsilon
    return P / denom


def one_hot_mode_distribution_loss(p: torch.Tensor) -> torch.Tensor:
    """
    L = -sum_k p_k^2 (Schur convexity / purity), minimized when p is one-hot.
    Mean over batch dimension if batch > 1.
    """
    return -(p**2).sum(dim=0).mean()


def optimize_tm_slm(
    tm_system: TMMeasurementSystem,
    slm_patterns: torch.Tensor,
    intensity_targets: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    planes: str = "both",
    optimize_aberrations: bool = True,
    test_data=None,
    use_exposure_normalization: bool = False,
):
    """Notebook-equivalent optimize_tm_slm logic from TM_retrieve.ipynb."""
    print("Starting Adam Optimization using TMMeasurementSystem internal parameters.")
    print(f"Optimize Aberrations: {optimize_aberrations}")

    tm_system._optimized_tm_param.requires_grad_(True)
    tm_system.slm_complex_amplitude.requires_grad_(True)
    for param in [
        tm_system.zernike_before_ft,
        tm_system.zernike_after_ft,
        tm_system.zernike_image_plane,
        tm_system.zernike_fourier_domain,
    ]:
        param.requires_grad_(optimize_aberrations)
    for param_dict in [tm_system.zoomfft_slm_params, tm_system.zoomfft_out_fwd_params, tm_system.zoomfft_out_inv_params]:
        for param in param_dict.values():
            param.requires_grad_(optimize_aberrations)
    # Keep polarization-angle aberration params consistent with optimize_aberrations flag.
    # On some PyTorch/CPU builds, letting these real params receive complex-chain grads
    # can trigger a dtype mismatch during backward.
    if hasattr(tm_system, "theta_before_slm"):
        tm_system.theta_before_slm.requires_grad_(optimize_aberrations)
    if hasattr(tm_system, "theta_after_slm"):
        tm_system.theta_after_slm.requires_grad_(optimize_aberrations)

    def _is_aberration(name):
        return ("zernike" in name or "zoomfft" in name or name in ("theta_before_slm", "theta_after_slm"))

    aberration_params = [p for n, p in tm_system.named_parameters() if p.requires_grad and _is_aberration(n)]
    slm_complex_params = [p for n, p in tm_system.named_parameters() if p.requires_grad and n == "slm_complex_amplitude"]
    other_params = [p for n, p in tm_system.named_parameters() if p.requires_grad and not _is_aberration(n) and n != "slm_complex_amplitude"]

    optimizer = optim.Adam(
        [
            {"params": other_params, "lr": learning_rate},
            {"params": aberration_params, "lr": learning_rate / 50},
            {"params": slm_complex_params, "lr": learning_rate / 2},
        ]
    )
    num_trainable_params = sum(p.numel() for p in tm_system.parameters() if p.requires_grad)
    print(f"Number of parameters to be optimized: {num_trainable_params}")
    criterion = nn.MSELoss()
    criterion = PearsonCorrelationLoss(dim=(0, 2, 3), center=True)
    if not isinstance(planes, str):
        raise TypeError(f"planes must be a string, got {type(planes)}: {planes}")
    planes_l = planes.lower()
    num_measurements = slm_patterns.shape[0]
    num_batches = (num_measurements + batch_size - 1) // batch_size
    history = {"epoch_losses": [], "test_losses": [], "tm_norm": []}
    print(f"Measurements: {num_measurements} | Batches/Epoch: {num_batches} | LR: {learning_rate}")

    if not isinstance(num_epochs, int):
        num_epochs = int(num_epochs)


    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        indices = torch.randperm(num_measurements)
        for i in range(num_batches):
            optimizer.zero_grad()
            start = i * batch_size
            end = min(start + batch_size, num_measurements)
            batch_idx = indices[start:end]
            pred_intensity = tm_system.forward_model_slm(slm_patterns[batch_idx])
            if use_exposure_normalization:
                with torch.no_grad():
                    numerator = torch.sum(pred_intensity * intensity_targets[batch_idx], dim=(-1, -2), keepdim=True)
                    denominator = torch.sum(pred_intensity**2, dim=(-1, -2), keepdim=True)
                    scale = (numerator / denominator).mean(dim=0, keepdim=True).detach()
            else:
                scale = 1.0

            pred_normalized = pred_intensity * scale
            if planes_l == "image":
                pred_normalized = pred_normalized[:, 0:1]
                target_normalized = intensity_targets[batch_idx][:, 0:1]
            elif planes_l == "fourier":
                pred_normalized = pred_normalized[:, 1:2]
                target_normalized = intensity_targets[batch_idx][:, 1:2]
            else:
                target_normalized = intensity_targets[batch_idx]

            if i == 0:
                with torch.no_grad():
                    pred_power = torch.sum(pred_intensity, dim=(-1, -2)).mean(dim=0)
                    rel_power = pred_power[0] / pred_power[1] * (tm_system.dx["im"] / tm_system.dx["four"]) ** 2
                print(f"rel_power: {rel_power}")

            pred_amp, target_amp = pred_normalized, target_normalized
            norm_factor = torch.mean(target_amp).detach()
            norm_factor = 1.0
            loss = criterion(pred_amp / norm_factor, target_amp / norm_factor) + 1e-5 * tm_system.slm_complex_amplitude.abs().sum()
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / num_batches
        test_loss = None
        if test_data is not None:
            with torch.no_grad():
                test_slm = test_data["slm_patterns"].to(tm_system.device)
                test_targets = test_data["intensity_targets"].to(tm_system.device)
                test_pred = tm_system.forward_model_slm(test_slm)
                if use_exposure_normalization:
                    if planes_l == "image":
                        test_pred_sel = test_pred[:, 0:1]
                        test_target_sel = test_targets[:, 0:1]
                        numerator_test = torch.sum(test_pred_sel * test_target_sel)
                        denominator_test = torch.sum(test_pred_sel**2)
                        scale_test = (numerator_test / denominator_test).detach()
                        test_pred_normalized = test_pred_sel * scale_test
                        test_target_normalized = test_target_sel
                    elif planes_l == "fourier":
                        test_pred_sel = test_pred[:, 1:2]
                        test_target_sel = test_targets[:, 1:2]
                        numerator_test = torch.sum(test_pred_sel * test_target_sel)
                        denominator_test = torch.sum(test_pred_sel**2)
                        scale_test = (numerator_test / denominator_test).detach()
                        test_pred_normalized = test_pred_sel * scale_test
                        test_target_normalized = test_target_sel
                    else:
                        test_pred_image = test_pred[:, 0:1]
                        test_target_image = test_targets[:, 0:1]
                        numerator_img = torch.sum(test_pred_image * test_target_image)
                        denominator_img = torch.sum(test_pred_image**2)
                        scale_img = (numerator_img / denominator_img).detach()
                        test_pred_fourier = test_pred[:, 1:2]
                        test_target_fourier = test_targets[:, 1:2]
                        numerator_fourier = torch.sum(test_pred_fourier * test_target_fourier)
                        denominator_fourier = torch.sum(test_pred_fourier**2)
                        scale_fourier = (numerator_fourier / denominator_fourier).detach()
                        test_pred_normalized = torch.cat([test_pred_image * scale_img, test_pred_fourier * scale_fourier], dim=1)
                        test_target_normalized = test_targets
                else:
                    if planes_l == "image":
                        test_pred_normalized = test_pred[:, 0:1]
                        test_target_normalized = test_targets[:, 0:1]
                    elif planes_l == "fourier":
                        test_pred_normalized = test_pred[:, 1:2]
                        test_target_normalized = test_targets[:, 1:2]
                    else:
                        test_pred_normalized = test_pred
                        test_target_normalized = test_targets
                test_pred_amp = torch.sqrt(test_pred_normalized)
                test_target_amp = torch.sqrt(test_target_normalized)
                norm_factor_test = torch.mean(test_target_amp).detach()
                test_loss = criterion(test_pred_amp / norm_factor_test, test_target_amp / norm_factor_test).item()

        with torch.no_grad():
            tm_norm = torch.norm(tm_system.optimized_tm).item()
        history["epoch_losses"].append(avg_loss)
        if test_loss is not None:
            history["test_losses"].append(test_loss)
        history["tm_norm"].append(tm_norm)
        if test_loss is not None:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_loss:.6f}, Test Loss={test_loss:.6f}, TM Norm={tm_norm:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}, TM Norm={tm_norm:.6f}")

    print("Optimization Completed.")
    return history


@dataclass
class PipelineConfig:
    n_samples: int = 6000
    n_out: int = 128
    ratio: int = 5
    sleep_time: float = 0.06

    slm_orig_lib: bool = True
    cameras_sn: Tuple[str, str, str, str] = ("40237702", "40238131", "40238137", "40237686")
    transmission_image_et: Tuple[float, ...] = (120.0,)
    transmission_fourier_et: Tuple[float, ...] = (150.0,)
    gain: float = 9.0

    train_test_split_ratio: float = 0.95
    random_seed: int = 42
    device: str = "cuda"
    optimize_aberrations: bool = True
    optimize_planes: str = "both"
    tm_num_epochs: int = 100
    tm_learning_rate: float = 0.01
    tm_batch_size: int = 256
    use_exposure_normalization: bool = False

    slm_opt_iters: int = 150
    slm_opt_lr: float = 0.1
    focus_center_radius_px: int = 1
    target_plane: str = "image"
    measured_focus_window: int = 2
    target_max_offset_from_center_px: int = 30

    output_h5_path: str = "measurement_focus.h5"
    output_group: str = "experiment_0"
    output_dir: str = "paper/image_results"
    tm_checkpoint_path: str = "tm_focus_outputs/tm_system_checkpoint.pt"
    one_spot_repeats: int = 5
    multispot_count: int = 3
    multispot_min_distance_px: int = 15
    multispot_balance_weight_image: float = 10.0
    multispot_balance_weight_fourier: float = 100.0
    target_image_names: Tuple[str, str] = ("target_1.png", "target_2.png")
    target_image_inner_size: int = 64

    run_one_hot_mode_opt: bool = True
    one_hot_mode_opt_iters: int = 150
    one_hot_mode_opt_lr: float = 0.1
    one_hot_power_epsilon: float = 1e-12
    # Random SLM patterns used to report the TM-predicted fraction of power in the
    # target mode before optimization (mean/std over patterns). 0 disables.
    one_hot_random_baseline_patterns: int = 100


class TMFocusPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.names = ["transmission_image", "transmission_fourier", "PhaseMask"]
        self.exposure_times_vector = [
            [cfg.transmission_image_et[0] * cfg.gain],
            [cfg.transmission_fourier_et[0] * cfg.gain],
        ]
        self.values = np.round(np.linspace(0, 255, 256)).astype(np.uint8)
        self.pm_center_x = int(np.load("PM_center_x.npy").item())
        self.pm_center_y = int(np.load("PM_center_y.npy").item())
        self._load_geometry()

        self.slm_lib = None
        self.cameras = None

    def _load_geometry(self) -> None:
        with open("centers_dict.pkl", "rb") as f:
            centers_dict = pickle.load(f)

        dx_cam = 3.45e-6
        self.magnification = 50.0
        self.dx_im = {"tran": dx_cam / self.magnification, "ref": dx_cam / self.magnification}
        self.dx_four = {"tran": dx_cam, "ref": dx_cam * (250.0 / 200.0)}
        na = 0.22
        margin = FIBER_CROP_MARGIN
        obj_f = 200e-3 / self.magnification
        core_radius = 25e-6
        # Kept for the enhancement-factor support (fiber core / NA disk).
        self.na = na
        self.obj_f = obj_f
        self.core_radius = core_radius
        self.dx_slm = 8e-6 * (150.0 / 400.0)
        four_range = margin * na * obj_f
        im_range = margin * core_radius
        self.n_pix = {"tran": {}, "ref": {}, "SLM": 320}
        for side in ["tran", "ref"]:
            self.n_pix[side]["im"] = round(im_range / self.dx_im[side])
            self.n_pix[side]["four"] = round(four_range / self.dx_four[side])

        self.n_pix_map = {
            "transmission_image": self.n_pix["tran"]["im"],
            "transmission_fourier": self.n_pix["tran"]["four"],
            "PhaseMask": self.n_pix["SLM"],
        }
        self.x_centers = {
            "transmission_image": centers_dict["TI"][0],
            "transmission_fourier": centers_dict["TF"][0],
            "PhaseMask": self.pm_center_x,
        }
        self.y_centers = {
            "transmission_image": centers_dict["TI"][1],
            "transmission_fourier": centers_dict["TF"][1],
            "PhaseMask": self.pm_center_y,
        }

    def support_radius_px(self, plane: str, out_size: int = None) -> float:
        """
        Radius, in pixels of the cropped-and-resized frame, of the region the fiber can
        illuminate: the core (radius a) in the image plane, the NA disk (radius NA * f_obj)
        in the Fourier plane. Same geometry as crop_and_resize_image and the TM grid.
        """
        out = int(self.cfg.n_out if out_size is None else out_size)
        if plane.lower() == "image":
            phys_radius, dx, crop_half = self.core_radius, self.dx_im["tran"], self.n_pix["tran"]["im"]
        elif plane.lower() == "fourier":
            phys_radius, dx, crop_half = self.na * self.obj_f, self.dx_four["tran"], self.n_pix["tran"]["four"]
        else:
            raise ValueError(f"plane must be 'image' or 'fourier', got {plane!r}")
        # crop_and_resize_image maps 2 * crop_half camera pixels onto `out` pixels.
        return (phys_radius / dx) * out / (2.0 * crop_half)

    def support_mask(self, plane: str, shape: Tuple[int, int]) -> np.ndarray:
        """Boolean fiber-core (image) or NA-disk (Fourier) mask for a frame of `shape`."""
        return disk_support_mask(shape, self.support_radius_px(plane, out_size=shape[0]))

    def support_mask_torch(self, plane: str, shape: Tuple[int, int], device) -> torch.Tensor:
        return torch.from_numpy(self.support_mask(plane, shape)).to(device)

    def setup(self) -> None:
        if self.cfg.slm_orig_lib:
            self.slm_lib, _, _, _, _, _ = create_SLM()
        else:
            self.slm_lib, _, _ = create_SLMpy()
        self.cameras = create_cmeras(
            list(self.cfg.cameras_sn),
            [
                [self.exposure_times_vector[0][0]],
                [self.exposure_times_vector[1][0]],
                [self.exposure_times_vector[0][0]],
                [self.exposure_times_vector[1][0]],
            ],
        )

    def teardown(self) -> None:
        if self.cameras is not None:
            for cam in self.cameras:
                cam.Close()
        if self.slm_lib is not None:
            if self.cfg.slm_orig_lib:
                self.slm_lib.Delete_SDK()
            else:
                self.slm_lib.close()

    def capture_hdr_images(self, camera, exposure_times):
        image = []
        for exposure_time in exposure_times:
            camera.ExposureTime.SetValue(exposure_time)
            if camera.WaitForFrameTriggerReady(200, py.TimeoutHandling_ThrowException):
                camera.ExecuteSoftwareTrigger()
            grab_result = camera.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
            image.append(grab_result.Array)
        return image if len(image) > 1 else image[0]

    def capture_transmission_images(self):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.capture_hdr_images, cam, et)
                for cam, et in zip(self.cameras[:2], self.exposure_times_vector)
            ]
            return [future.result() for future in futures]

    def project_phase_mask_to_slm(self, phase_mask: np.ndarray):
        if self.cfg.slm_orig_lib:
            self.slm_lib.Write_image(phase_mask.flatten().ctypes.data_as(POINTER(c_ubyte)), c_uint(1))
        else:
            self.slm_lib.updateArray(phase_mask)
        time.sleep(self.cfg.sleep_time)

    def crop_and_resize_image(self, image: np.ndarray, name: str):
        x_center = self.x_centers[name]
        y_center = self.y_centers[name]
        crop_size = self.n_pix_map[name]
        cropped = image[y_center - crop_size:y_center + crop_size, x_center - crop_size:x_center + crop_size]
        resized = cv2.resize(cropped, (self.cfg.n_out, self.cfg.n_out), interpolation=cv2.INTER_LINEAR)
        return resized

    def save_image_to_h5(self, image, name: str, h5_group):
        if isinstance(image, list) and len(image) > 1:
            processed_images = [self.crop_and_resize_image(img, name) for img in image]
            im0 = np.round(processed_images[0]).astype(np.uint16)
            im1 = np.round(processed_images[1]).astype(np.uint16)
            saturated = im1 == 255
            im1[saturated] = im0[saturated] * 5
            final_image = im1
        else:
            if isinstance(image, list):
                image = image[0]
            final_image = np.round(self.crop_and_resize_image(image, name)).astype(np.uint8)

        if name in h5_group:
            dataset = h5_group[name]
            current_size = dataset.shape[0]
            dataset.resize(current_size + 1, axis=0)
            dataset[current_size] = final_image
        else:
            maxshape = (None, self.cfg.n_out, self.cfg.n_out)
            h5_group.create_dataset(name, data=[final_image], maxshape=maxshape, chunks=True)

    def save_images_to_h5(self, images: List[np.ndarray], names: List[str], h5_group):
        with ThreadPoolExecutor() as executor:
            for image, name in zip(images, names):
                executor.submit(self.save_image_to_h5, np.squeeze(np.array(image)), name, h5_group)

    def generate_random_phase_mask(self) -> np.ndarray:
        small = np.zeros((1200 // self.cfg.ratio, 1920 // self.cfg.ratio), dtype=np.uint8)
        crop_size = self.n_pix["SLM"]
        y_start_full = self.pm_center_y - crop_size
        y_end_full = self.pm_center_y + crop_size
        x_start_full = self.pm_center_x - crop_size
        x_end_full = self.pm_center_x + crop_size
        y_start = max(0, y_start_full // self.cfg.ratio)
        y_end = min(small.shape[0], (y_end_full + self.cfg.ratio - 1) // self.cfg.ratio)
        x_start = max(0, x_start_full // self.cfg.ratio)
        x_end = min(small.shape[1], (x_end_full + self.cfg.ratio - 1) // self.cfg.ratio)
        random_array = np.random.choice(self.values, size=(y_end - y_start, x_end - x_start)).astype(np.uint8)
        small[y_start:y_end, x_start:x_end] = random_array
        return np.kron(small, np.ones((self.cfg.ratio, self.cfg.ratio), dtype=np.uint8))

    def sample_transmission_data(self) -> str:
        if Path(self.cfg.output_h5_path).exists():
            with h5py.File(self.cfg.output_h5_path, "r") as h5f:
                if self.cfg.output_group in h5f:
                    h5_group = h5f[self.cfg.output_group]
                    required = all(k in h5_group for k in self.names)
                    if required:
                        n_existing = int(h5_group["PhaseMask"].shape[0])
                        print(
                            f"Existing sampled file found ({self.cfg.output_h5_path}, {self.cfg.output_group}) "
                            f"with {n_existing} samples. Reusing it and skipping sampling."
                        )
                        return self.cfg.output_h5_path

        np.random.seed(self.cfg.random_seed)
        with h5py.File(self.cfg.output_h5_path, "a") as h5f:
            if self.cfg.output_group not in h5f:
                h5_group = h5f.create_group(self.cfg.output_group)
            else:
                h5_group = h5f[self.cfg.output_group]

            blank = np.zeros((1200, 1920), dtype=np.uint8)
            self.project_phase_mask_to_slm(blank)

            for _ in tqdm(range(self.cfg.n_samples), desc="Sampling transmission data"):
                phase_mask = self.generate_random_phase_mask()
                self.project_phase_mask_to_slm(phase_mask)
                captured = self.capture_transmission_images()
                captured.append(phase_mask)
                self.save_images_to_h5(captured, self.names, h5_group)
        return self.cfg.output_h5_path

    def initialize_tm_system(self):
        wavelength = 632.8e-9
        core_index = 1.457
        na = 0.22
        core_radius = 25e-6
        cladding_index = np.sqrt(core_index**2 - na**2)
        n0 = self.cfg.n_out
        obj_f = 200e-3 / self.magnification

        dx_final = {}
        for side in ["tran", "ref"]:
            dx_final[side] = {
                "im": self.dx_im[side] * self.n_pix[side]["im"] / 64.0,
                "four": self.dx_four[side] * self.n_pix[side]["four"] / 64.0,
                "slm": self.dx_slm * self.n_pix["SLM"] / 64.0,
            }

        mult_factor = {
            "tran": dx_final["tran"]["im"] ** 2 / (wavelength * obj_f),
            "slm": dx_final["tran"]["slm"] ** 2 / (wavelength * obj_f),
        }
        fiber_params = {
            "a": core_radius,
            "n1": core_index,
            "n0": cladding_index,
            "wavelength": wavelength,
        }

        range_im = (n0 / 2) * dx_final["tran"]["im"]
        grid_params = {
            "x_range": (-range_im, range_im),
            "y_range": (-range_im, range_im),
            "nx": n0,
            "ny": n0,
        }

        a_param = {
            "slm": 2 * np.pi * dx_final["tran"]["slm"] * dx_final["tran"]["im"] / (wavelength * obj_f),
            "tran": 2 * np.pi * dx_final["tran"]["im"] * dx_final["tran"]["four"] / (wavelength * obj_f),
        }
        zoomfft_params = {
            side: {
                "a_init": a_param[side],
                "n0_init": (n0 - 1) / 2,
                "m0_init": (n0 - 1) / 2,
                "k0_init": (n0 - 1) / 2,
                "l0_init": (n0 - 1) / 2,
            }
            for side in ["slm", "tran"]
        }
        runtime_device = self.cfg.device
        if runtime_device == "cuda" and not torch.cuda.is_available():
            runtime_device = "cpu"

        tm_system = TMMeasurementSystem(
            fiber_params,
            grid_params,
            device=runtime_device,
            mult_factor=mult_factor,
            zoomfft_params=zoomfft_params,
            zernike_n_modes=8,
            dx=dx_final["tran"],
            cache_dir=str(REPO_ROOT / "data" / "mode_fields"),
        )
        return tm_system

    def optimize_slm_pattern_with_tm(
        self,
        tm_system,
        slm_shape=(128, 128),
        num_iterations=100,
        lr=0.1,
        center_radius=1,
        target_plane="image",
        target_yx: Tuple[int, int] = None,
    ):
        for param in tm_system.parameters():
            param.requires_grad = False
        tm_system.eval()

        device = tm_system.device
        slm_phase = nn.Parameter(2 * torch.pi * torch.rand((1, *slm_shape), dtype=torch.float32, device=device))
        optimizer = optim.Adam([slm_phase], lr=lr)
        h = tm_system.grid_params["ny"]
        w = tm_system.grid_params["nx"]
        if target_yx is None:
            cy, cx = h // 2, w // 2
        else:
            cy, cx = int(target_yx[0]), int(target_yx[1])
        center_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
        y_min, y_max = max(0, cy - center_radius), min(h, cy + center_radius + 1)
        x_min, x_max = max(0, cx - center_radius), min(w, cx + center_radius + 1)
        center_mask[y_min:y_max, x_min:x_max] = True
        plane_idx = 0 if target_plane.lower() == "image" else 1
        plane = "image" if plane_idx == 0 else "fourier"
        background_mask = self.support_mask_torch(plane, (h, w), device) & ~center_mask

        history = []
        for i in range(num_iterations):
            optimizer.zero_grad()
            pred_intensity = tm_system.forward_model_slm(slm_phase)
            plane_intensity = pred_intensity[0, plane_idx]
            center_mean = plane_intensity[center_mask].mean()
            background_mean = plane_intensity[background_mask].mean()
            focus_metric = center_mean / background_mean
            loss = -focus_metric
            loss.backward()
            optimizer.step()
            history.append(float(focus_metric.item()))
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}/{num_iterations} | Focus Metric: {focus_metric.item():.2f}")
        optimized_phase = torch.remainder(slm_phase.detach(), 2 * torch.pi).squeeze(0)
        return optimized_phase, history

    @staticmethod
    def phase_to_uint8(phase_rad: np.ndarray) -> np.ndarray:
        phase_wrapped = np.mod(phase_rad, 2.0 * np.pi)
        return np.floor(phase_wrapped * (256.0 / (2.0 * np.pi))).astype(np.uint8)

    def pack_phase_to_full_slm(self, phase_uint8_small: np.ndarray) -> np.ndarray:
        full_small = np.zeros((1200 // self.cfg.ratio, 1920 // self.cfg.ratio), dtype=np.uint8)
        crop_size = self.n_pix["SLM"]
        y_start_full = self.pm_center_y - crop_size
        y_end_full = self.pm_center_y + crop_size
        x_start_full = self.pm_center_x - crop_size
        x_end_full = self.pm_center_x + crop_size
        y_start = max(0, y_start_full // self.cfg.ratio)
        y_end = min(full_small.shape[0], (y_end_full + self.cfg.ratio - 1) // self.cfg.ratio)
        x_start = max(0, x_start_full // self.cfg.ratio)
        x_end = min(full_small.shape[1], (x_end_full + self.cfg.ratio - 1) // self.cfg.ratio)
        h = min(y_end - y_start, phase_uint8_small.shape[0])
        w = min(x_end - x_start, phase_uint8_small.shape[1])
        full_small[y_start:y_start + h, x_start:x_start + w] = np.flip(phase_uint8_small[:h, :w])
        return np.kron(full_small, np.ones((self.cfg.ratio, self.cfg.ratio), dtype=np.uint8))

    def center_background_focus_metric(
        self,
        image_2d: np.ndarray,
        center_radius: int,
        target_yx: Tuple[int, int] = None,
        plane: str = "image",
    ) -> float:
        """Enhancement factor: target-window mean over the mean of the fiber support
        (core in the image plane, NA disk in the Fourier plane) outside the window."""
        h, w = image_2d.shape
        if target_yx is None:
            cy, cx = h // 2, w // 2
        else:
            cy, cx = int(target_yx[0]), int(target_yx[1])
        y0, y1 = max(0, cy - center_radius), min(h, cy + center_radius + 1)
        x0, x1 = max(0, cx - center_radius), min(w, cx + center_radius + 1)
        target_mask = np.zeros((h, w), dtype=bool)
        target_mask[y0:y1, x0:x1] = True
        return enhancement_factor(image_2d, target_mask, self.support_mask(plane, (h, w)))

    def sample_target_pixel_near_center(self, image_shape: Tuple[int, int], max_offset: int) -> Tuple[int, int]:
        h, w = image_shape
        cy, cx = h // 2, w // 2
        while True:
            dy = np.random.randint(-max_offset, max_offset + 1)
            dx = np.random.randint(-max_offset, max_offset + 1)
            if dy * dy + dx * dx <= max_offset * max_offset:
                ty = int(np.clip(cy + dy, 0, h - 1))
                tx = int(np.clip(cx + dx, 0, w - 1))
                return ty, tx

    def sample_distinct_targets_near_center(
        self,
        image_shape: Tuple[int, int],
        max_offset: int,
        count: int,
        min_distance: int,
    ) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        tries = 0
        while len(points) < count and tries < 5000:
            tries += 1
            cand = self.sample_target_pixel_near_center(image_shape, max_offset)
            if all((cand[0] - p[0]) ** 2 + (cand[1] - p[1]) ** 2 >= min_distance**2 for p in points):
                points.append(cand)
        if len(points) < count:
            raise RuntimeError("Could not sample distinct points with requested spacing.")
        return points

    def multi_spot_focus_metric(
        self,
        image_2d: np.ndarray,
        center_radius: int,
        target_points: List[Tuple[int, int]],
        plane: str = "image",
    ) -> float:
        """Mean of the per-spot means over the mean of the fiber support outside all spots."""
        h, w = image_2d.shape
        spot_mask = np.zeros((h, w), dtype=bool)
        spot_means = []
        for ty, tx in target_points:
            y0, y1 = max(0, ty - center_radius), min(h, ty + center_radius + 1)
            x0, x1 = max(0, tx - center_radius), min(w, tx + center_radius + 1)
            spot_mask[y0:y1, x0:x1] = True
            spot_means.append(float(image_2d[y0:y1, x0:x1].mean()))
        background_mask = self.support_mask(plane, (h, w)) & ~spot_mask
        background = image_2d[background_mask]
        return float(np.mean(spot_means) / background.mean())

    def optimize_slm_pattern_multispot_with_tm(
        self,
        tm_system,
        target_points: List[Tuple[int, int]],
        slm_shape=(128, 128),
        num_iterations=150,
        lr=0.1,
        center_radius=1,
        target_plane="image",
        balance_weight=0.1,
    ):
        for param in tm_system.parameters():
            param.requires_grad = False
        tm_system.eval()

        device = tm_system.device
        slm_phase = nn.Parameter(2 * torch.pi * torch.rand((1, *slm_shape), dtype=torch.float32, device=device))
        optimizer = optim.Adam([slm_phase], lr=lr)
        h = tm_system.grid_params["ny"]
        w = tm_system.grid_params["nx"]
        plane_idx = 0 if target_plane.lower() == "image" else 1
        plane = "image" if plane_idx == 0 else "fourier"

        spot_masks = []
        occupied = torch.zeros((h, w), dtype=torch.bool, device=device)
        for ty, tx in target_points:
            mask = torch.zeros((h, w), dtype=torch.bool, device=device)
            y0, y1 = max(0, ty - center_radius), min(h, ty + center_radius + 1)
            x0, x1 = max(0, tx - center_radius), min(w, tx + center_radius + 1)
            mask[y0:y1, x0:x1] = True
            spot_masks.append(mask)
            occupied |= mask
        background_mask = self.support_mask_torch(plane, (h, w), device) & ~occupied

        history = []
        for i in range(num_iterations):
            optimizer.zero_grad()
            pred_intensity = tm_system.forward_model_slm(slm_phase)
            plane_intensity = pred_intensity[0, plane_idx]
            spot_means = torch.stack([plane_intensity[m].mean() for m in spot_masks])
            mean_spot = spot_means.mean()
            background_mean = plane_intensity[background_mask].mean()
            focus_metric = mean_spot / background_mean
            balance_penalty = torch.std(spot_means) / (mean_spot + 1e-9)
            loss = -focus_metric + balance_weight * balance_penalty
            loss.backward()
            optimizer.step()
            history.append(float(focus_metric.item()))
            if (i + 1) % 50 == 0:
                print(f"[multispot] Iteration {i + 1}/{num_iterations} | Focus Metric: {focus_metric.item():.2f}")

        optimized_phase = torch.remainder(slm_phase.detach(), 2 * torch.pi).squeeze(0)
        return optimized_phase, history

    def build_centered_target_from_image(self, image_path: Path, inner_size: int, canvas_size: int = 128) -> np.ndarray:
        target_src = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if target_src is None:
            raise FileNotFoundError(f"Could not read target image: {image_path}")
        resized = cv2.resize(target_src, (inner_size, inner_size), interpolation=cv2.INTER_AREA).astype(np.float32)
        centered = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        y0 = (canvas_size - inner_size) // 2
        x0 = (canvas_size - inner_size) // 2
        centered[y0:y0 + inner_size, x0:x0 + inner_size] = resized
        return centered

    def optimize_slm_pattern_correlation_with_target(
        self,
        tm_system,
        target_image: np.ndarray,
        slm_shape=(128, 128),
        num_iterations=150,
        lr=0.1,
        target_plane="image",
    ):
        for param in tm_system.parameters():
            param.requires_grad = False
        tm_system.eval()

        device = tm_system.device
        target_t = torch.tensor(target_image, dtype=torch.float32, device=device)
        target_t = (target_t - target_t.mean()) / (target_t.std() + 1e-8)
        slm_phase = nn.Parameter(2 * torch.pi * torch.rand((1, *slm_shape), dtype=torch.float32, device=device))
        optimizer = optim.Adam([slm_phase], lr=lr)
        plane_idx = 0 if target_plane.lower() == "image" else 1
        history = []

        for i in range(num_iterations):
            optimizer.zero_grad()
            pred_intensity = tm_system.forward_model_slm(slm_phase)[0, plane_idx]
            pred_norm = (pred_intensity - pred_intensity.mean()) / (pred_intensity.std() + 1e-8)
            corr = torch.mean(pred_norm * target_t)
            loss = -corr
            loss.backward()
            optimizer.step()
            history.append(float(corr.item()))
            if (i + 1) % 50 == 0:
                print(f"[target-corr] Iteration {i + 1}/{num_iterations} | Corr: {corr.item():.4f}")

        optimized_phase = torch.remainder(slm_phase.detach(), 2 * torch.pi).squeeze(0)
        return optimized_phase, history

    def optimize_slm_pattern_one_hot_modes_with_tm(
        self,
        tm_system: TMMeasurementSystem,
        slm_shape: Tuple[int, int] = (128, 128),
        num_iterations: int = 150,
        lr: float = 0.1,
        power_normalize_epsilon: float = 1e-12,
    ) -> Tuple[torch.Tensor, List[float], dict]:
        """
        Optimize SLM phase so the *output* guided-mode distribution is nearly one-hot:
        p_k = P_k / (sum_j P_j + eps) with P_k = |c_k|^2 from slm_to_output_modes, and
        minimize L = -sum_k p_k^2.

        Does not write camera images; project the returned phase with ``pack_phase_to_full_slm``
        + ``project_phase_mask_to_slm`` and capture from the transmission cameras in ``run``.
        """
        for param in tm_system.parameters():
            param.requires_grad = False
        tm_system.eval()

        device = tm_system.device
        slm_phase = nn.Parameter(2 * torch.pi * torch.rand((1, *slm_shape), dtype=torch.float32, device=device))
        optimizer = optim.Adam([slm_phase], lr=lr)
        history: List[float] = []

        # Mode-power distribution at the random initial SLM phase (before optimization).
        with torch.no_grad():
            p_initial = output_modes_to_normalized_powers(
                tm_system.slm_to_output_modes(slm_phase), epsilon=power_normalize_epsilon
            )[:, 0].detach().cpu().numpy()

        for i in range(num_iterations):
            optimizer.zero_grad()
            out_modes = tm_system.slm_to_output_modes(slm_phase)
            p = output_modes_to_normalized_powers(out_modes, epsilon=power_normalize_epsilon)
            loss = one_hot_mode_distribution_loss(p)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                purity = (p**2).sum(dim=0).mean().item()
            history.append(purity)
            if (i + 1) % 50 == 0:
                print(
                    f"[one-hot modes] Iteration {i + 1}/{num_iterations} | "
                    f"sum_k p_k^2 (higher→more concentrated): {purity:.6f}"
                )

        optimized_phase = torch.remainder(slm_phase.detach(), 2 * torch.pi).squeeze(0)

        summary: dict = {}
        with torch.no_grad():
            out_modes_final = tm_system.slm_to_output_modes(slm_phase)
            p_final = output_modes_to_normalized_powers(out_modes_final, epsilon=power_normalize_epsilon)
            summary["p_final"] = p_final[:, 0].detach().cpu().numpy()
            summary["argmax_mode"] = int(p_final[:, 0].argmax().item())
            summary["max_p"] = float(p_final[:, 0].max().item())
            k = summary["argmax_mode"]
            summary["p_initial"] = p_initial
            # Fraction of power in the target mode k at the initial (random) SLM phase.
            summary["p_initial_target_mode"] = float(p_initial[k])
            summary["purity_initial"] = float((p_initial**2).sum())

            # Same fraction over an ensemble of independent random SLM patterns.
            n_random = int(getattr(self.cfg, "one_hot_random_baseline_patterns", 0))
            if n_random > 0:
                p_k_random = []
                chunk = 8
                for start in range(0, n_random, chunk):
                    b = min(chunk, n_random - start)
                    rand_phase = 2 * torch.pi * torch.rand((b, *slm_shape), dtype=torch.float32, device=device)
                    p_rand = output_modes_to_normalized_powers(
                        tm_system.slm_to_output_modes(rand_phase), epsilon=power_normalize_epsilon
                    )
                    p_k_random.append(p_rand[k, :].detach().cpu().numpy())
                p_k_random = np.concatenate(p_k_random)
                summary["p_random_target_mode_mean"] = float(p_k_random.mean())
                summary["p_random_target_mode_std"] = float(p_k_random.std())
                summary["n_random_patterns"] = int(p_k_random.size)

        print(
            f"[one-hot modes] target mode {k}: p_initial={summary['p_initial_target_mode']:.4%} "
            f"-> p_final={summary['max_p']:.4%}"
        )
        return optimized_phase, history, summary

    def capture_transmission_one_hot_modes_pattern(self, optimized_phase_rad: torch.Tensor) -> None:
        """Project optimized one-hot SLM phase and capture image + Fourier cameras (same crop/flip as main pipeline)."""
        optimized_phase_np = optimized_phase_rad.detach().cpu().numpy()
        optimized_phase_uint8 = self.phase_to_uint8(optimized_phase_np)
        full_phase_mask_uint8 = self.pack_phase_to_full_slm(optimized_phase_uint8)
        self.project_phase_mask_to_slm(full_phase_mask_uint8)
        np.save(self.output_dir / "projected_full_phase_mask_uint8_one_hot_modes.npy", full_phase_mask_uint8)

        one_hot_capture_planes = [
            ("image", 0, "transmission_image", True, float(self.cfg.transmission_image_et[0])),
            ("fourier", 1, "transmission_fourier", False, float(self.cfg.transmission_fourier_et[0])),
        ]
        for plane_name, camera_idx, crop_name, should_flip, exposure_us in one_hot_capture_planes:
            et = exposure_us * self.cfg.gain
            raw = self.capture_hdr_images(self.cameras[camera_idx], [et])
            raw = np.squeeze(np.array(raw))
            proc = self.crop_and_resize_image(raw, crop_name).astype(np.float32)
            if should_flip:
                proc = np.flip(proc)
            cv2.imwrite(
                str(self.output_dir / f"im_one_hot_modes_{plane_name}.png"),
                np.clip(proc, 0, 255).astype(np.uint8),
            )
            np.save(self.output_dir / f"measured_{crop_name}_one_hot_modes.npy", proc)

    def run(self):
        self.setup()
        try:
            print("Step 1/6: Sampling transmission-only data")
            h5_path = self.sample_transmission_data()

            print("Step 2/6: Preprocessing + parameter initialization")
            et_ratio = self.cfg.transmission_image_et[0] / self.cfg.transmission_fourier_et[0]
            bs_split = 0.47736
            tran_im_factor = self.magnification**2 * bs_split / (1 - bs_split) * et_ratio
            train_data, test_data = load_real_mmf_data(
                h5_file_path=h5_path,
                experiment_group=self.cfg.output_group,
                train_test_split_ratio=self.cfg.train_test_split_ratio,
                random_seed=self.cfg.random_seed,
                im_factor=tran_im_factor,
            )
            tm_system = self.initialize_tm_system()
            train_data["slm_patterns"] = train_data["slm_patterns"].to(tm_system.device)
            train_data["intensity_targets"] = train_data["intensity_targets"].to(tm_system.device)
            test_data["slm_patterns"] = test_data["slm_patterns"].to(tm_system.device)
            test_data["intensity_targets"] = test_data["intensity_targets"].to(tm_system.device)

            print("Step 3/6: Optimizing transmission matrix")
            tm_checkpoint = Path(self.cfg.tm_checkpoint_path)
            tm_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if tm_checkpoint.exists():
                print(f"TM checkpoint found at {tm_checkpoint}. Loading and skipping TM optimization.")
                tm_system = torch.load(tm_checkpoint, map_location=tm_system.device, weights_only=False)
                tm_history = {"loaded_from_checkpoint": True}
            else:
                tm_history = optimize_tm_slm(
                    tm_system=tm_system,
                    slm_patterns=train_data["slm_patterns"],
                    intensity_targets=train_data["intensity_targets"],
                    num_epochs=self.cfg.tm_num_epochs,
                    learning_rate=self.cfg.tm_learning_rate,
                    batch_size=self.cfg.tm_batch_size,
                    planes=self.cfg.optimize_planes,
                    optimize_aberrations=self.cfg.optimize_aberrations,
                    test_data=test_data,
                    use_exposure_normalization=self.cfg.use_exposure_normalization,
                )
                torch.save(tm_system, tm_checkpoint)
                print(f"Saved TM checkpoint to {tm_checkpoint}")

            one_hot_mode_summary = None
            one_hot_mode_history = None
            if self.cfg.run_one_hot_mode_opt:
                print(
                    "Optional step: SLM optimization for one-hot output mode distribution, "
                    "then SLM projection + camera capture (image and Fourier)"
                )
                oh_phase, one_hot_mode_history, one_hot_mode_summary = self.optimize_slm_pattern_one_hot_modes_with_tm(
                    tm_system=tm_system,
                    slm_shape=(self.cfg.n_out, self.cfg.n_out),
                    num_iterations=self.cfg.one_hot_mode_opt_iters,
                    lr=self.cfg.one_hot_mode_opt_lr,
                    power_normalize_epsilon=self.cfg.one_hot_power_epsilon,
                )
                np.save(self.output_dir / "optimized_phase_rad_one_hot_modes.npy", oh_phase.detach().cpu().numpy())
                self.capture_transmission_one_hot_modes_pattern(oh_phase)

            one_spot_results_by_plane = {}
            multispot_results_by_plane = {}
            target_correlation_results_by_plane = {}
            plane_configs = [
                (
                    "image",
                    0,
                    "transmission_image",
                    True,
                    float(self.cfg.transmission_image_et[0]),
                    float(self.cfg.multispot_balance_weight_image),
                ),
                (
                    "fourier",
                    1,
                    "transmission_fourier",
                    False,
                    float(self.cfg.transmission_fourier_et[0]),
                    float(self.cfg.multispot_balance_weight_fourier),
                ),
            ]

            for plane_name, camera_idx, crop_name, should_flip, exposure_time, plane_balance_weight in plane_configs:
                print(f"Step 4-7/6: Running experiments for {plane_name} plane")
                one_spot_results = []
                for i in range(self.cfg.one_spot_repeats):
                    target_yx = self.sample_target_pixel_near_center(
                        (self.cfg.n_out, self.cfg.n_out),
                        self.cfg.target_max_offset_from_center_px,
                    )
                    print(
                        f"Step 4/6 [{plane_name}] [{i+1}/{self.cfg.one_spot_repeats}]: "
                        f"One-point optimize at (y={target_yx[0]}, x={target_yx[1]})"
                    )
                    optimized_phase, slm_opt_history = self.optimize_slm_pattern_with_tm(
                        tm_system=tm_system,
                        slm_shape=(self.cfg.n_out, self.cfg.n_out),
                        num_iterations=self.cfg.slm_opt_iters,
                        lr=self.cfg.slm_opt_lr,
                        center_radius=self.cfg.focus_center_radius_px,
                        target_plane=plane_name,
                        target_yx=target_yx,
                    )

                    optimized_phase_np = optimized_phase.detach().cpu().numpy()
                    optimized_phase_uint8 = self.phase_to_uint8(optimized_phase_np)
                    full_phase_mask_uint8 = self.pack_phase_to_full_slm(optimized_phase_uint8)
                    self.project_phase_mask_to_slm(full_phase_mask_uint8)

                    transmission_img = self.capture_hdr_images(self.cameras[camera_idx], [20])
                    transmission_img = np.squeeze(np.array(transmission_img))
                    transmission_img_proc = self.crop_and_resize_image(transmission_img, crop_name).astype(np.float32)
                    if should_flip:
                        transmission_img_proc = np.flip(transmission_img_proc)
                    measured_focus = self.center_background_focus_metric(
                        transmission_img_proc,
                        center_radius=self.cfg.measured_focus_window,
                        target_yx=target_yx,
                        plane=plane_name,
                    )
                    cv2.imwrite(
                        str(self.output_dir / f"im_{plane_name}_{i+1}.png"),
                        np.clip(transmission_img_proc, 0, 255).astype(np.uint8),
                    )
                    one_spot_results.append(
                        {
                            "target_pixel_yx": [int(target_yx[0]), int(target_yx[1])],
                            "final_measured_focus_metric": float(measured_focus),
                            "slm_opt_last_focus_metric_pred": slm_opt_history[-1] if slm_opt_history else None,
                        }
                    )
                one_spot_results_by_plane[plane_name] = one_spot_results

                print(f"Step 5/6 [{plane_name}]: Multi-spot optimization with 3 distinct points")
                multispot_targets = self.sample_distinct_targets_near_center(
                    (self.cfg.n_out, self.cfg.n_out),
                    self.cfg.target_max_offset_from_center_px,
                    self.cfg.multispot_count,
                    self.cfg.multispot_min_distance_px,
                )
                print(f"[{plane_name}] Selected multispot targets: {multispot_targets}")
                multi_phase, multi_history = self.optimize_slm_pattern_multispot_with_tm(
                    tm_system=tm_system,
                    target_points=multispot_targets,
                    slm_shape=(self.cfg.n_out, self.cfg.n_out),
                    num_iterations=self.cfg.slm_opt_iters,
                    lr=self.cfg.slm_opt_lr,
                    center_radius=self.cfg.focus_center_radius_px,
                    target_plane=plane_name,
                    balance_weight=plane_balance_weight,
                )
                multi_phase_np = multi_phase.detach().cpu().numpy()
                multi_phase_uint8 = self.phase_to_uint8(multi_phase_np)
                multi_full_phase_mask_uint8 = self.pack_phase_to_full_slm(multi_phase_uint8)
                self.project_phase_mask_to_slm(multi_full_phase_mask_uint8)

                print(f"Step 6/6 [{plane_name}]: Measuring multispot intensity")
                multi_img = self.capture_hdr_images(self.cameras[camera_idx], [30])
                multi_img = np.squeeze(np.array(multi_img))
                multi_img_proc = self.crop_and_resize_image(multi_img, crop_name).astype(np.float32)
                if should_flip:
                    multi_img_proc = np.flip(multi_img_proc)
                multi_focus = self.multi_spot_focus_metric(
                    multi_img_proc,
                    center_radius=self.cfg.measured_focus_window,
                    target_points=multispot_targets,
                    plane=plane_name,
                )
                cv2.imwrite(
                    str(self.output_dir / f"im_multispot_{plane_name}.png"),
                    np.clip(multi_img_proc, 0, 255).astype(np.uint8),
                )

                np.save(self.output_dir / f"optimized_phase_rad_multispot_{plane_name}.npy", multi_phase_np)
                np.save(self.output_dir / f"optimized_phase_uint8_multispot_{plane_name}.npy", multi_phase_uint8)
                np.save(
                    self.output_dir / f"projected_full_phase_mask_uint8_multispot_{plane_name}.npy",
                    multi_full_phase_mask_uint8,
                )
                np.save(self.output_dir / f"measured_{crop_name}_multispot_{plane_name}.npy", multi_img_proc)
                multispot_results_by_plane[plane_name] = {
                    "targets_yx": [[int(y), int(x)] for y, x in multispot_targets],
                    "balance_weight": float(plane_balance_weight),
                    "final_measured_focus_metric": float(multi_focus),
                    "slm_opt_last_focus_metric_pred": multi_history[-1] if multi_history else None,
                }

                print(f"Step 7/6 [{plane_name}]: Target-image correlation experiments")
                target_correlation_results = []
                for target_name in self.cfg.target_image_names:
                    target_stem = Path(target_name).stem
                    target_path = self.output_dir / target_name
                    print(f"[target-corr][{plane_name}] optimizing for {target_path}")
                    target_canvas = self.build_centered_target_from_image(
                        image_path=target_path,
                        inner_size=self.cfg.target_image_inner_size,
                        canvas_size=self.cfg.n_out,
                    )
                    cv2.imwrite(
                        str(self.output_dir / f"target_canvas_{target_stem}_{plane_name}.png"),
                        np.clip(target_canvas, 0, 255).astype(np.uint8),
                    )
                    corr_phase, corr_history = self.optimize_slm_pattern_correlation_with_target(
                        tm_system=tm_system,
                        target_image=target_canvas,
                        slm_shape=(self.cfg.n_out, self.cfg.n_out),
                        num_iterations=self.cfg.slm_opt_iters,
                        lr=self.cfg.slm_opt_lr,
                        target_plane=plane_name,
                    )
                    corr_phase_np = corr_phase.detach().cpu().numpy()
                    corr_phase_uint8 = self.phase_to_uint8(corr_phase_np)
                    corr_full_phase_mask_uint8 = self.pack_phase_to_full_slm(corr_phase_uint8)
                    self.project_phase_mask_to_slm(corr_full_phase_mask_uint8)
                    exposure_time = 640 if plane_name == "fourier" else 320
                    corr_img = self.capture_hdr_images(self.cameras[camera_idx], [exposure_time])
                    corr_img = np.squeeze(np.array(corr_img))
                    corr_img_proc = self.crop_and_resize_image(corr_img, crop_name).astype(np.float32)
                    if should_flip:
                        corr_img_proc = np.flip(corr_img_proc)
                    corr_img_norm = (corr_img_proc - corr_img_proc.mean()) / (corr_img_proc.std() + 1e-8)
                    target_norm = (target_canvas - target_canvas.mean()) / (target_canvas.std() + 1e-8)
                    measured_corr = float(np.mean(corr_img_norm * target_norm))
                    cv2.imwrite(
                        str(self.output_dir / f"im_target_corr_{target_stem}_{plane_name}.png"),
                        np.clip(corr_img_proc, 0, 255).astype(np.uint8),
                    )
                    np.save(
                        self.output_dir / f"optimized_phase_rad_target_corr_{target_stem}_{plane_name}.npy",
                        corr_phase_np,
                    )
                    np.save(
                        self.output_dir / f"optimized_phase_uint8_target_corr_{target_stem}_{plane_name}.npy",
                        corr_phase_uint8,
                    )
                    np.save(
                        self.output_dir / f"projected_full_phase_mask_uint8_target_corr_{target_stem}_{plane_name}.npy",
                        corr_full_phase_mask_uint8,
                    )
                    np.save(
                        self.output_dir / f"measured_{crop_name}_target_corr_{target_stem}_{plane_name}.npy",
                        corr_img_proc,
                    )
                    target_correlation_results.append(
                        {
                            "target_image_path": str(target_path),
                            "target_inner_size": self.cfg.target_image_inner_size,
                            "measured_correlation": measured_corr,
                            "slm_opt_last_corr_pred": corr_history[-1] if corr_history else None,
                        }
                    )
                target_correlation_results_by_plane[plane_name] = target_correlation_results
            with open(self.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
                one_hot_json = None
                if one_hot_mode_summary is not None:
                    one_hot_json = {
                        "argmax_mode": one_hot_mode_summary["argmax_mode"],
                        "max_p": one_hot_mode_summary["max_p"],
                        "p_final": one_hot_mode_summary["p_final"].tolist(),
                        "p_initial_target_mode": one_hot_mode_summary["p_initial_target_mode"],
                        "purity_initial": one_hot_mode_summary["purity_initial"],
                        "p_initial": one_hot_mode_summary["p_initial"].tolist(),
                        "p_random_target_mode_mean": one_hot_mode_summary.get("p_random_target_mode_mean"),
                        "p_random_target_mode_std": one_hot_mode_summary.get("p_random_target_mode_std"),
                        "n_random_patterns": one_hot_mode_summary.get("n_random_patterns", 0),
                        "purity_history_last": one_hot_mode_history[-1] if one_hot_mode_history else None,
                    }
                json.dump(
                    {
                        "enhancement_definition": ENHANCEMENT_DEFINITION,
                        "support_radius_px": {
                            "image": self.support_radius_px("image"),
                            "fourier": self.support_radius_px("fourier"),
                        },
                        "config": asdict(self.cfg),
                        "one_spot_results_by_plane": one_spot_results_by_plane,
                        "multispot_by_plane": multispot_results_by_plane,
                        "target_correlation_results_by_plane": target_correlation_results_by_plane,
                        "tm_history_keys": sorted(list(tm_history.keys())) if isinstance(tm_history, dict) else [],
                        "one_hot_mode_summary": one_hot_json,
                    },
                    f,
                    indent=2,
                )
            print("Done. Completed one-spot, multispot, and target-correlation on image + fourier planes.")
        finally:
            self.teardown()


def main():
    cfg = PipelineConfig()
    pipeline = TMFocusPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
