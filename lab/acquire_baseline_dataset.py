"""
Sample transmission data for TM calibration.

Camera images are saved at 128x128. SLM phase masks are saved at 32x32 with a
coarser kron upsampling (ratio 20) to the physical SLM aperture. No motor
initialization, movement, or initial calibration group.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))            # this repository's core modules
sys.path.insert(0, str(REPO_ROOT.parent))     # the private lab repo providing create_refs

from run_wavefront_control import (
    PipelineConfig,
    TMFocusPipeline,
    _resolve_output_h5_path,
    _resolve_tm_lib_path,
)


@dataclass
class RafDataAcquiringConfig(PipelineConfig):
    """Sampling config: 128x128 images, 32x32 SLM macro-grid."""

    n_out: int = 128
    slm_n_out: int = 32
    slm_ratio: int = field(init=False, default=20)

    output_h5_path: str = "data/measurement_raf.h5"
    output_group: str = "experiment_0"
    output_dir: str = "raf_data_outputs"
    force_resample: bool = False

    def __post_init__(self) -> None:
        if self.n_out <= 0 or self.slm_n_out <= 0:
            raise ValueError("n_out and slm_n_out must be positive")
        if self.slm_active_extent_px % self.slm_n_out != 0:
            raise ValueError(
                f"slm_n_out ({self.slm_n_out}) must divide "
                f"slm_active_extent_px ({self.slm_active_extent_px})"
            )
        if self.slm_active_extent_px % self.n_out != 0:
            raise ValueError(
                f"n_out ({self.n_out}) must divide "
                f"slm_active_extent_px ({self.slm_active_extent_px})"
            )
        self.slm_ratio = self.slm_active_extent_px // self.slm_n_out
        self.ratio = self.slm_active_extent_px // self.n_out
        self.output_h5_path = _resolve_output_h5_path(self.output_h5_path)
        self.output_dir = _resolve_tm_lib_path(self.output_dir)


class RafDataAcquiringPipeline(TMFocusPipeline):
    def __init__(self, cfg: RafDataAcquiringConfig):
        super().__init__(cfg)
        self.cfg: RafDataAcquiringConfig = cfg

    def _slm_macro_slice_bounds(self) -> Tuple[int, int, int, int]:
        ratio = self.cfg.slm_ratio
        n = self.cfg.slm_n_out
        cy = round(self.pm_center_y / ratio)
        cx = round(self.pm_center_x / ratio)
        y_start = cy - n // 2
        x_start = cx - n // 2
        return y_start, y_start + n, x_start, x_start + n

    def generate_random_phase_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        ratio = self.cfg.slm_ratio
        n = self.cfg.slm_n_out
        small = np.zeros((1200 // ratio, 1920 // ratio), dtype=np.uint8)
        y_start, y_end, x_start, x_end = self._slm_macro_slice_bounds()
        if y_start < 0 or x_start < 0 or y_end > small.shape[0] or x_end > small.shape[1]:
            raise ValueError(
                f"SLM macro crop ({y_start}:{y_end}, {x_start}:{x_end}) is out of bounds "
                f"for downsampled SLM shape {small.shape}"
            )
        random_array = np.random.choice(self.values, size=(n, n)).astype(np.uint8)
        small[y_start:y_end, x_start:x_end] = random_array
        full_mask = np.kron(small, np.ones((ratio, ratio), dtype=np.uint8))
        macro_grid = small[y_start:y_end, x_start:x_end].copy()
        assert macro_grid.shape == (n, n)
        return full_mask, macro_grid

    def save_phase_macro_to_h5(self, phase_macro: np.ndarray, h5_group) -> None:
        if phase_macro.shape != (self.cfg.slm_n_out, self.cfg.slm_n_out):
            raise ValueError(
                f"Expected PhaseMask shape {(self.cfg.slm_n_out, self.cfg.slm_n_out)}, "
                f"got {phase_macro.shape}"
            )
        final_image = np.round(phase_macro).astype(np.uint8)
        name = "PhaseMask"
        if name in h5_group:
            dataset = h5_group[name]
            current_size = dataset.shape[0]
            dataset.resize(current_size + 1, axis=0)
            dataset[current_size] = final_image
        else:
            maxshape = (None, self.cfg.slm_n_out, self.cfg.slm_n_out)
            h5_group.create_dataset(name, data=[final_image], maxshape=maxshape, chunks=True)

    def sample_transmission_data(self) -> str:
        h5_path = Path(self.cfg.output_h5_path)
        if not self.cfg.force_resample and h5_path.exists():
            with h5py.File(h5_path, "r") as h5f:
                if self.cfg.output_group in h5f:
                    h5_group = h5f[self.cfg.output_group]
                    required = all(k in h5_group for k in self.names)
                    if required:
                        n_existing = int(h5_group["PhaseMask"].shape[0])
                        if n_existing >= self.cfg.n_samples:
                            print(
                                f"Existing sampled file found ({self.cfg.output_h5_path}, "
                                f"{self.cfg.output_group}) with {n_existing} samples. "
                                "Reusing and skipping sampling."
                            )
                            return self.cfg.output_h5_path

        np.random.seed(self.cfg.random_seed)
        with h5py.File(self.cfg.output_h5_path, "a") as h5f:
            if self.cfg.force_resample and self.cfg.output_group in h5f:
                del h5f[self.cfg.output_group]
            if self.cfg.output_group not in h5f:
                h5_group = h5f.create_group(self.cfg.output_group)
            else:
                h5_group = h5f[self.cfg.output_group]

            blank = np.zeros((1200, 1920), dtype=np.uint8)
            self.project_phase_mask_to_slm(blank)

            for _ in tqdm(range(self.cfg.n_samples), desc="Sampling transmission data"):
                phase_mask, phase_macro = self.generate_random_phase_mask()
                self.project_phase_mask_to_slm(phase_mask)
                captured = self.capture_transmission_images()
                self.save_images_to_h5(captured, self.names[:2], h5_group)
                self.save_phase_macro_to_h5(phase_macro, h5_group)
        return self.cfg.output_h5_path

    def run(self) -> Dict:
        self.setup()
        summary: Dict = {
            "n_samples": self.cfg.n_samples,
            "image_resolution": self.cfg.n_out,
            "slm_resolution": self.cfg.slm_n_out,
            "slm_ratio": self.cfg.slm_ratio,
            "output_h5_path": self.cfg.output_h5_path,
            "output_group": self.cfg.output_group,
        }
        try:
            print(
                f"Sampling {self.cfg.n_samples} measurements "
                f"({self.cfg.n_out}x{self.cfg.n_out} images, "
                f"{self.cfg.slm_n_out}x{self.cfg.slm_n_out} SLM, "
                f"kron ratio {self.cfg.slm_ratio})"
            )
            self.sample_transmission_data()
            summary["config"] = asdict(self.cfg)
            summary_path = self.output_dir / "raf_sampling_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Done. Summary: {summary_path}")
            print(f"Measurements saved to: {self.cfg.output_h5_path}")
            return summary
        finally:
            self.teardown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample transmission data (128x128 images, 32x32 SLM masks)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4 * 1508,
        help="Number of samples (default: PipelineConfig n_samples)",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="data/measurement_raf.h5",
        help="H5 file path (default: data/measurement_raf.h5)",
    )
    parser.add_argument(
        "--output-group",
        type=str,
        default="experiment_0",
        help="H5 group name (default: experiment_0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="raf_data_outputs",
        help="Output directory for summary JSON (default: raf_data_outputs)",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Overwrite existing H5 group instead of reusing it",
    )

    args = parser.parse_args()

    cfg = RafDataAcquiringConfig()
    if args.n_samples > 0:
        cfg.n_samples = args.n_samples
    if args.output_h5:
        cfg.output_h5_path = args.output_h5
    if args.output_group:
        cfg.output_group = args.output_group
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.force_resample:
        cfg.force_resample = True

    pipeline = RafDataAcquiringPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
