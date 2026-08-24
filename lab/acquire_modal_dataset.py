"""
Experimental pipeline: sample transmission data at the current motor position,
move the motor to a new position, and sample again.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))            # this repository's core modules
sys.path.insert(0, str(REPO_ROOT.parent))     # the private lab repo providing create_refs

from create_refs import connect_motor, set_camera_pixel_format
from motor_pos_utils import CALIBRATION_H5_GROUP, home_motor
from run_wavefront_control import PipelineConfig, TMFocusPipeline

MEASUREMENT_PIXEL_FORMAT = "Mono12"
DEFAULT_PIXEL_FORMAT = "Mono8"
CAMERA_IMAGE_NAMES = frozenset({"transmission_image", "transmission_fourier"})


@dataclass
class MotorSamplingConfig(PipelineConfig):
    """Configuration for two-position motor displacement sampling."""

    gain: float = 1.0
    camera_pixel_format: str = MEASUREMENT_PIXEL_FORMAT
    initial_h5_group: str = CALIBRATION_H5_GROUP
    post_motor_h5_group: str = "post_motor"
    max_motor_len: float = 10.0
    motor_position_mm: float = 5.0
    motor_settle_time: float = 5.0
    home_settle_time: float = 10.0
    force_resample: bool = False
    from_step: int = 1
    output_dir: str = "results/offline/modal"
    output_h5_path: str = "data/measurement_motor_pcc.h5"
    use_optimized_patterns: bool = True
    optimized_masks_h5_path: str = "results/offline/modal/spectral_init_slm_masks.h5"
    optimized_masks_group: str = "mode_excitation_patterns"


class MotorSamplingPipeline(TMFocusPipeline):
    def __init__(self, cfg: MotorSamplingConfig):
        super().__init__(cfg)
        self.cfg: MotorSamplingConfig = cfg
        self.motor = None
        self.exposure_times_vector = [
            list(cfg.transmission_image_et),
            list(cfg.transmission_fourier_et),
        ]

    def setup(self) -> None:
        super().setup()
        for cam_idx, exposure_times in enumerate(self.exposure_times_vector[:2]):
            set_camera_pixel_format(
                self.cameras[cam_idx],
                self.cfg.camera_pixel_format,
                exposure_time=exposure_times[0],
            )

    def teardown(self) -> None:
        if self.cameras is not None:
            for cam in self.cameras[:2]:
                set_camera_pixel_format(cam, DEFAULT_PIXEL_FORMAT)
        super().teardown()

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
            processed = self.crop_and_resize_image(image, name)
            if name in CAMERA_IMAGE_NAMES:
                final_image = np.round(processed).astype(np.uint16)
            else:
                final_image = np.round(processed).astype(np.uint8)

        if name in h5_group:
            dataset = h5_group[name]
            current_size = dataset.shape[0]
            dataset.resize(current_size + 1, axis=0)
            dataset[current_size] = final_image
        else:
            maxshape = (None, self.cfg.n_out, self.cfg.n_out)
            h5_group.create_dataset(name, data=[final_image], maxshape=maxshape, chunks=True)

    def _load_optimized_small_masks(self) -> np.ndarray:
        if getattr(self, "_optimized_masks", None) is not None:
            return self._optimized_masks
        import h5py

        path = Path(self.cfg.optimized_masks_h5_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Optimized SLM masks not found at {path}. Generate them first with "
                "lab/run_wavefront_control.py (one-hot mode excitation)."
            )
        with h5py.File(path, "r") as h5f:
            group = self.cfg.optimized_masks_group
            if group not in h5f:
                raise KeyError(f"Group '{group}' not in {path}. Available: {list(h5f.keys())}")
            if "phase_masks_small_uint8" not in h5f[group]:
                raise KeyError(f"'phase_masks_small_uint8' missing in group '{group}' of {path}.")
            masks = np.array(h5f[group]["phase_masks_small_uint8"][:], dtype=np.uint8)
        print(f"Loaded {masks.shape[0]} optimized SLM patterns from {path} (group '{group}')")
        self._optimized_masks = masks
        return masks

    def sample_transmission_data_to_group(
        self,
        n_samples: int,
        group_name: str,
        *,
        force: bool = False,
    ) -> str:
        if self.cfg.use_optimized_patterns:
            optimized = self._load_optimized_small_masks()
            target_count = int(optimized.shape[0])
        else:
            optimized = None
            target_count = int(n_samples)

        h5_path = Path(self.cfg.output_h5_path)
        if not force and h5_path.exists():
            import h5py

            with h5py.File(h5_path, "r") as h5f:
                if group_name in h5f:
                    h5_group = h5f[group_name]
                    if all(k in h5_group for k in self.names):
                        n_existing = int(h5_group["PhaseMask"].shape[0])
                        if n_existing >= target_count:
                            print(
                                f"Reusing existing group '{group_name}' "
                                f"({n_existing} samples >= {target_count})."
                            )
                            return str(h5_path)

        import h5py

        kind = "optimized" if optimized is not None else "random"
        print(f"Sampling {target_count} {kind} measurements into group '{group_name}'")
        with h5py.File(self.cfg.output_h5_path, "a") as h5f:
            if group_name in h5f and force:
                del h5f[group_name]
            if group_name not in h5f:
                h5_group = h5f.create_group(group_name)
            else:
                h5_group = h5f[group_name]

            blank = np.zeros((1200, 1920), dtype=np.uint8)
            self.project_phase_mask_to_slm(blank)

            for i in tqdm(range(target_count), desc=f"Sampling {group_name}"):
                if optimized is not None:
                    phase_mask = self.pack_phase_to_full_slm(optimized[i])
                else:
                    phase_mask = self.generate_random_phase_mask()
                self.project_phase_mask_to_slm(phase_mask)
                captured = self.capture_transmission_images()
                captured.append(phase_mask)
                self.save_images_to_h5(captured, self.names, h5_group)
        return self.cfg.output_h5_path

    def run(self) -> Dict:
        self.setup()
        summary: Dict = {
            "motor_location_mm": None,
            "n_samples": self.cfg.n_samples,
            "initial_h5_group": self.cfg.initial_h5_group,
            "post_motor_h5_group": self.cfg.post_motor_h5_group,
            "output_h5_path": self.cfg.output_h5_path,
        }
        try:
            n_samples = self.cfg.n_samples
            print(f"Sampling {n_samples} measurements per position")

            if self.cfg.from_step <= 1:
                print("Step 1/3: Homing motor and sampling calibration")
                self.motor = connect_motor()
                home_motor(self.motor)
                print(f"Motor homed; settling {self.cfg.home_settle_time}s")
                time.sleep(self.cfg.home_settle_time)
                self.sample_transmission_data_to_group(
                    n_samples,
                    self.cfg.initial_h5_group,
                    force=self.cfg.force_resample,
                )
            else:
                print(
                    f"Skipping step 1; reusing initial group '{self.cfg.initial_h5_group}'"
                )
                if self.motor is None:
                    self.motor = connect_motor()

            if self.cfg.from_step <= 2:
                motor_loc = float(self.cfg.motor_position_mm)
                if not (1e-2 <= motor_loc <= self.cfg.max_motor_len):
                    raise ValueError(
                        f"motor_position_mm ({motor_loc}) must be in "
                        f"[0.01, max_motor_len={self.cfg.max_motor_len}]"
                    )
                print(f"Step 2/3: Moving motor to {motor_loc:.4f} mm")
                self.motor.move_to(motor_loc, blocking=True)
                summary["motor_location_mm"] = float(motor_loc)
                print(
                    f"Motor moved to {motor_loc:.4f} mm; "
                    f"settling {self.cfg.motor_settle_time}s"
                )
                time.sleep(self.cfg.motor_settle_time)
            else:
                summary["motor_location_mm"] = float(self.cfg.motor_position_mm)
                print(
                    f"Skipping step 2 (motor move); "
                    f"assuming post-motor data in '{self.cfg.post_motor_h5_group}'"
                )

            if self.cfg.from_step <= 3:
                print("Step 3/3: Sampling at new motor position")
                self.sample_transmission_data_to_group(
                    n_samples,
                    self.cfg.post_motor_h5_group,
                    force=self.cfg.force_resample,
                )
            else:
                print(
                    f"Skipping step 3; reusing post-motor group "
                    f"'{self.cfg.post_motor_h5_group}'"
                )

            summary["config"] = asdict(self.cfg)
            summary_path = self.output_dir / "motor_sampling_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Done. Summary: {summary_path}")
            print(f"Measurements saved to: {self.cfg.output_h5_path}")
            return summary
        finally:
            self.teardown()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample transmission data at two motor positions"
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Resume from this step (1=full run, 2=skip initial sampling, 3=post-motor only)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="Number of samples per position (default: PipelineConfig n_samples)",
    )
    parser.add_argument(
        "--motor-position-mm",
        type=float,
        default=None,
        help="Target motor position in mm for the second sampling",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Overwrite existing H5 groups instead of reusing them",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for summary JSON (default: results/offline/modal)",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="",
        help="H5 file path for measurements (default: data/measurement_motor_pcc.h5)",
    )
    parser.add_argument(
        "--use-optimized-patterns",
        action="store_true",
        help="Project pre-optimized diversity-maximizing SLM masks instead of random ones",
    )
    parser.add_argument(
        "--random-patterns",
        action="store_true",
        help="Project freshly randomized SLM masks instead of the optimized ones",
    )
    parser.add_argument(
        "--optimized-masks-h5",
        type=str,
        default="results/offline/modal/spectral_init_slm_masks.h5",
        help="H5 file with optimized masks",
    )
    parser.add_argument(
        "--optimized-masks-group",
        type=str,
        default="",
        help="Group in the optimized-masks H5 (default: mode_excitation_patterns)",
    )

    args = parser.parse_args()

    cfg = MotorSamplingConfig()
    cfg.from_step = args.from_step
    if args.n_samples > 0:
        cfg.n_samples = args.n_samples
    if args.motor_position_mm is not None:
        cfg.motor_position_mm = args.motor_position_mm
    if args.force_resample:
        cfg.force_resample = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.output_h5:
        cfg.output_h5_path = args.output_h5
    if args.random_patterns:
        cfg.use_optimized_patterns = False
    elif args.use_optimized_patterns:
        cfg.use_optimized_patterns = True
    if args.optimized_masks_h5:
        cfg.optimized_masks_h5_path = args.optimized_masks_h5
    if args.optimized_masks_group:
        cfg.optimized_masks_group = args.optimized_masks_group

    pipeline = MotorSamplingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
