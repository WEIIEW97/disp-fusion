"""Experimental native, fixed-cost backend; never replaces V1 or V2."""
import ctypes as ct
from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path

import numpy as np

from .pipeline import RefinementResult


@dataclass(frozen=True)
class FastAlignedConfig:
    tile_size: int = 16
    plane_radius_cells: int = 1
    alignment_radius_cells: int = 2
    boundary_guard_px: int = 2
    edge_band_px: int = 6
    minimum_anchors: int = 64
    minimum_votes: int = 3
    minimum_coverage: float = .7
    mono_edge: float = .012
    stereo_jump_px: float = 2.
    anchor_deviation_px: float = .75
    plane_rms_px: float = .35
    mono_plane_rms: float = .002
    affine_rms_px: float = 1.
    minimum_mono_std: float = .005
    reject_threshold_px: float = 4.
    vote_fraction: float = .8
    smooth_strength: float = .85
    max_change_px: float = .75
    edge_mode: int = 1
    edge_window: int = 49
    edge_stride: int = 16
    edge_samples: int = 64
    edge_iterations: int = 3
    edge_min_samples: int = 12
    edge_anchor_guard_px: int = 3
    edge_fit_p80_px: float = 2.
    edge_min_mono_span: float = .025
    edge_scale_floor_px: float = .15

    def __post_init__(self):
        for field in fields(self):
            if field.type is not int:
                continue
            if type(getattr(self, field.name)) is not int:
                raise ValueError(f'{field.name} must be an integer')
        if self.tile_size not in (8, 16, 32):
            raise ValueError('tile_size must be 8, 16 or 32')
        if not 1 <= self.plane_radius_cells <= 4 or not 1 <= self.alignment_radius_cells <= 8:
            raise ValueError('invalid model-window radius')
        if not 0 <= self.boundary_guard_px <= 16 or not self.boundary_guard_px <= self.edge_band_px <= 32:
            raise ValueError('invalid boundary/edge-band radius')
        if self.minimum_anchors < 3 or not 1 <= self.minimum_votes <= 4:
            raise ValueError('invalid anchor/vote requirement')
        for name in ['mono_edge', 'stereo_jump_px', 'anchor_deviation_px', 'plane_rms_px',
                     'mono_plane_rms', 'affine_rms_px', 'minimum_mono_std', 'reject_threshold_px']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ['minimum_coverage', 'vote_fraction', 'smooth_strength', 'max_change_px']:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if not 0 < self.minimum_coverage <= 1 or not .5 < self.vote_fraction <= 1:
            raise ValueError('invalid coverage/vote fraction')
        if not 0 <= self.smooth_strength <= 1 or self.max_change_px < 0:
            raise ValueError('invalid smoothing strength/change bound')
        if self.edge_mode not in (0, 1, 2):
            raise ValueError('edge_mode must be 0 (legacy), 1 (robust edge first) or 2 (spatial consensus guard)')
        if not 8 <= self.edge_stride <= 64 or not 7 <= self.edge_window <= 4*self.edge_stride-1 or self.edge_window%2 != 1:
            raise ValueError('edge window must be odd and cover at most four grid centers per axis')
        if not 16 <= self.edge_samples <= 256 or not 1 <= self.edge_iterations <= 8:
            raise ValueError('invalid fixed edge sample/iteration budget')
        if not 4 <= self.edge_min_samples <= math.isqrt(self.edge_samples)**2 or not 0 <= self.edge_anchor_guard_px <= 16:
            raise ValueError('invalid edge sample support/anchor guard')
        for name in ['edge_fit_p80_px', 'edge_min_mono_span', 'edge_scale_floor_px']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive and finite')


class _Config(ct.Structure):
    _fields_ = [(f.name, ct.c_int32 if f.type is int else ct.c_float)
                for f in fields(FastAlignedConfig)]


class _Stats(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ['valid_pixels', 'anchor_pixels', 'boundary_pixels',
        'plane_models', 'alignment_models', 'removed_pixels', 'smoothed_pixels', 'uncertain_pixels',
        'workspace_bytes','edge_pixels','edge_covered_pixels','edge_attempted_windows',
        'edge_sampled_windows','edge_samples_total']]+[('mono_min', ct.c_float), ('mono_max', ct.c_float), ('phase_ms', ct.c_double*7)]


def _array(value, shape, dtype, *, writable=False):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f'expected a {dtype} NumPy array of shape {shape}; conversions are explicit')
    if not value.flags.c_contiguous or not value.flags.aligned or (writable and not value.flags.writeable):
        raise ValueError('aligned contiguous arrays and writable output buffers are required')
    return value.ctypes.data_as(ct.c_void_p)


class FastAlignedRefiner:
    """Float32 disparity and inverse-depth-like mono on the SAME pixel grid.

    Construction loads an explicitly built shared library and allocates native
    buffers once. process_into() reuses caller output buffers. A context cannot
    process concurrent frames; create a separate instance per stream.
    Spatial registration and neural inference are outside this backend.
    """
    def __init__(self, width, height, config=None, *, library=None):
        if type(width) is not int or type(height) is not int or not 2 <= width <= 8192 or not 2 <= height <= 8192:
            raise ValueError('width/height must be integers in [2, 8192]')
        self.width, self.height, self.shape = width, height, (height, width)
        self.config = config or FastAlignedConfig()
        if not isinstance(self.config, FastAlignedConfig):
            raise TypeError('config must be FastAlignedConfig')
        selected = library or os.environ.get('DISP_FUSION_NATIVE_LIB')
        if selected is None:
            selected = Path(__file__).resolve().parents[1]/'build/native/libdisp_fusion_fast.so'
        self.library_path = str(Path(selected).expanduser().resolve())
        if not Path(self.library_path).is_file():
            raise FileNotFoundError('Build the optional native backend with '
                'cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release and '
                'cmake --build build/native; or pass library=/path/to/libdisp_fusion_fast.so')
        self._lib = ct.CDLL(self.library_path)
        self._lib.df_fast_abi_version.restype = ct.c_uint32
        if self._lib.df_fast_abi_version() != 2:
            raise RuntimeError('unsupported native ABI version')
        self._lib.df_fast_build_features.restype = ct.c_uint32
        self.build_features = self._lib.df_fast_build_features()
        if self.config.edge_mode == 2 and not self.build_features & 32:
            raise RuntimeError('Rebuild the native library for discrete-neighbor edge_mode=2 support')
        self._lib.df_fast_create.argtypes = [ct.c_int32, ct.c_int32, ct.POINTER(_Config)]
        self._lib.df_fast_create.restype = ct.c_void_p
        self._lib.df_fast_destroy.argtypes = [ct.c_void_p]
        self._lib.df_fast_destroy.restype = None
        self._lib.df_fast_workspace_bytes.argtypes = [ct.c_void_p]
        self._lib.df_fast_workspace_bytes.restype = ct.c_size_t
        self._lib.df_fast_process.argtypes = [ct.c_void_p]*7+[ct.POINTER(_Stats)]
        self._lib.df_fast_process.restype = ct.c_int
        native = _Config(**asdict(self.config))
        self._context = self._lib.df_fast_create(width, height, ct.byref(native))
        if not self._context:
            raise RuntimeError('native configuration rejected or workspace allocation failed')

    @property
    def workspace_bytes(self):
        self._require_open()
        return self._lib.df_fast_workspace_bytes(self._context)

    def _require_open(self):
        if not getattr(self, '_context', None):
            raise RuntimeError('refiner is closed')

    def close(self):
        if getattr(self, '_context', None):
            self._lib.df_fast_destroy(self._context)
            self._context = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def process_into(self, stereo, mono, output, removed_mask, smoothed_mask, *, protected_mask=None):
        self._require_open()
        source = _array(stereo, self.shape, np.float32)
        guide = _array(mono, self.shape, np.float32)
        out = _array(output, self.shape, np.float32, writable=True)
        masks = []
        for value in [removed_mask, smoothed_mask]:
            if isinstance(value, np.ndarray) and value.dtype == np.bool_:
                value = value.view(np.uint8)
            masks.append(_array(value, self.shape, np.uint8, writable=True))
        if protected_mask is not None and isinstance(protected_mask, np.ndarray) and protected_mask.dtype == np.bool_:
            protected_mask = protected_mask.view(np.uint8)
        protected = None if protected_mask is None else _array(protected_mask, self.shape, np.uint8)
        stats = _Stats()
        status = self._lib.df_fast_process(self._context, source, guide, protected, out, *masks, ct.byref(stats))
        if status:
            raise ValueError('native input/output buffers overlap or arguments are invalid')
        return {name: list(getattr(stats, name)) if name == 'phase_ms' else getattr(stats, name)
                for name, _ in _Stats._fields_}

    def refine(self, stereo, mono, *, protected_mask=None):
        """Convenience allocation/export-compatible result; not the DSP hot path."""
        output = np.empty(self.shape, np.float32)
        removed = np.empty(self.shape, bool)
        smoothed = np.empty(self.shape, bool)
        stats = self.process_into(stereo, mono, output, removed, smoothed, protected_mask=protected_mask)
        rejected = stereo.copy()
        rejected[removed] = np.nan
        return RefinementResult('fast_aligned', output, removed, smoothed, rejected,
            dict(native=stats, configuration=asdict(self.config), build_features=self.build_features,
                 experimental=True))
