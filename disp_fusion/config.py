"""Version-independent settings for the shared V1/V2 pipeline."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class WindowConfig:
    window: int = 97
    stride: int = 32
    edge_radius: int = 6
    mono_gradient: float = .012
    stereo_gradient_px: float = 2.
    anchor_deviation_px: float = .75
    minimum_anchors: int = 60
    minimum_mono_span: float = .025
    maximum_fit_p80_px: float = 2.

    def __post_init__(self):
        for name in ['window', 'stride', 'edge_radius', 'minimum_anchors']:
            if not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        if self.window < 5 or self.window % 2 != 1 or self.stride < 1 or self.edge_radius < 0 or self.minimum_anchors < 3:
            raise ValueError('invalid window/stride/radius/anchor settings')
        for name in ['mono_gradient', 'stereo_gradient_px', 'anchor_deviation_px',
                     'minimum_mono_span', 'maximum_fit_p80_px']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class InteriorSmoothConfig:
    """Optional multi-scale affine-disparity smoothing inside frozen boundaries."""
    enabled: bool = False
    windows: tuple[int, ...] = (31, 63, 127)
    stride_fraction: float = .5
    strength: float = .85
    minimum_coverage: float = .7
    minimum_anchors: int = 128
    maximum_samples: int = 512
    stereo_fit_p90_px: float = .5
    mono_fit_p90: float = .003
    anchor_deviation_px: float = .75
    fit_iterations: int = 5
    boundary_guard_radius: int = 3
    guide_hole_radius: float = 1.5

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('interior enabled must be boolean')
        if (not self.windows or any(not isinstance(w, int) or w < 7 or w % 2 != 1 for w in self.windows)
                or len(set(self.windows)) != len(self.windows)):
            raise ValueError('interior windows must be unique odd integers >=7')
        object.__setattr__(self, 'windows', tuple(sorted(self.windows)))
        for name in ['stride_fraction', 'minimum_coverage']:
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f'interior {name} must lie in (0, 1]')
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError('interior strength must lie in [0, 1]')
        for name in ['minimum_anchors', 'maximum_samples', 'fit_iterations']:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 3:
                raise ValueError(f'interior {name} must be an integer >=3')
        if self.maximum_samples < self.minimum_anchors:
            raise ValueError('interior maximum_samples must be >= minimum_anchors')
        if not isinstance(self.boundary_guard_radius, int) or self.boundary_guard_radius < 0:
            raise ValueError('interior boundary_guard_radius must be a non-negative integer')
        if not math.isfinite(self.guide_hole_radius) or self.guide_hole_radius < 0:
            raise ValueError('interior guide_hole_radius must be non-negative and finite')
        for name in ['stereo_fit_p90_px', 'mono_fit_p90', 'anchor_deviation_px']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'interior {name} must be positive and finite')


@dataclass(frozen=True)
class RefinementConfig:
    alignment: WindowConfig = field(default_factory=WindowConfig)
    threshold_px: float = 4.
    minimum_votes: int = 3
    vote_fraction: float = .8
    smooth_strength: float = 4.
    smooth_iterations: int = 16
    mono_sigma: float = .015
    stereo_sigma_px: float = 1.
    stereo_jump_limit_px: float = 2.
    max_change_px: float = .75
    boundary_guard_radius: int = 3
    interior: InteriorSmoothConfig = field(default_factory=InteriorSmoothConfig)

    def __post_init__(self):
        if not isinstance(self.alignment, WindowConfig):
            raise ValueError('alignment must be WindowConfig')
        if not isinstance(self.interior, InteriorSmoothConfig):
            raise ValueError('interior must be InteriorSmoothConfig')
        for name in ['threshold_px', 'mono_sigma', 'stereo_sigma_px', 'stereo_jump_limit_px']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ['smooth_strength', 'max_change_px']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be non-negative and finite')
        for name in ['minimum_votes', 'smooth_iterations', 'boundary_guard_radius']:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if self.minimum_votes < 1 or not math.isfinite(self.vote_fraction) or not .5 < self.vote_fraction <= 1:
            raise ValueError('minimum_votes >=1 and .5 < vote_fraction <=1 required')
