"""Reusable non-neural V1/V2 refinement of stereo using cached mono structure."""
from .calibration import CameraIntrinsics, DepthRgbCalibration, load_depth_rgb_calibration
from .config import WindowConfig, InteriorSmoothConfig, RefinementConfig
from .pipeline import DepthRefiner, RefinementResult
from .fast_aligned import FastAlignedConfig, FastAlignedRefiner

__version__ = '1.0.0'
__all__ = ['CameraIntrinsics', 'DepthRgbCalibration', 'load_depth_rgb_calibration',
           'WindowConfig', 'InteriorSmoothConfig', 'RefinementConfig', 'DepthRefiner', 'RefinementResult',
           'FastAlignedConfig', 'FastAlignedRefiner']
