"""One processing pipeline; V1 and V2 share projection, alignment and export."""
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .alignment import align_windows, dilate, reject_from_rgb, valid_disparity
from .calibration import DepthRgbCalibration, project_depth_disparity_to_rgb
from .config import RefinementConfig
from .smoothing import bounded_guided_smooth, multiscale_interior_smooth


def consensus_mask(evidence, cfg=RefinementConfig()):
    """Add a consensus veto to V1's decision, so deletions are a V1 subset."""
    count = evidence['vote_count']
    agreement = np.maximum(evidence['positive_votes'], evidence['negative_votes'])
    fraction = agreement.astype(np.float32)/np.maximum(count, 1)
    old = (evidence['edge_band'] & np.isfinite(evidence['residual_px'])
           & (evidence['residual_px'] > cfg.threshold_px))
    selected = old & (count >= cfg.minimum_votes) & (fraction >= cfg.vote_fraction)
    return selected, fraction


def source_mask(original, source_index, rgb_mask):
    available = (source_index >= 0) & rgb_mask
    indices = source_index[available].astype(np.int64)
    if np.any(indices >= original.size):
        raise ValueError('source index exceeds original size')
    result = np.zeros(original.size, bool)
    result[indices] = True
    return result.reshape(original.shape) & valid_disparity(original)


def mapped_guide(original, source_index, evidence, old_removed, cfg=RefinementConfig()):
    """Map native mono structure to z-buffer source points, never fill holes."""
    seen = source_index >= 0
    source = source_index[seen].astype(np.int64)
    mono = np.full(original.size, np.nan, np.float32)
    mono[source] = evidence['mono_normalized'][seen]
    protected_rgb = dilate(evidence['mono_edge'], cfg.boundary_guard_radius)
    protected = np.zeros(original.size, bool)
    protected[source] = protected_rgb[seen]
    # V1-rejected but V2-restored pixels are uncertain: keep their original
    # value, instead of treating a failed consensus as evidence to smooth.
    safe = np.isfinite(mono).reshape(original.shape) & ~protected.reshape(original.shape) & ~old_removed
    return mono.reshape(original.shape), safe & valid_disparity(original)



@dataclass
class RefinementResult:
    version: str
    disparity: np.ndarray
    removed_mask: np.ndarray
    smoothed_mask: np.ndarray
    rejected_disparity: np.ndarray
    diagnostics: dict


class DepthRefiner:
    """Input disparity uses the stereo reference camera's calibrated coordinates.

    Project stereo into RGB, compare with native RGB mono, then map decisions
    and guidance back through the projection's source-pixel correspondence.
    Output has the same coordinates and orientation as the stereo input.
    No orientation conversion, inference, global data paths, or historical
    fusion outputs are accessed by this array-level API.
    """

    def __init__(self, calibration: DepthRgbCalibration, config: RefinementConfig | None = None):
        self.calibration = calibration
        self.config = config or RefinementConfig()

    def refine(self, stereo: np.ndarray, mono: np.ndarray, version: str = 'v2') -> RefinementResult:
        return self.refine_many(stereo, mono, (version,))[version.lower()]

    def refine_many(self, stereo: np.ndarray, mono: np.ndarray,
                    versions: Iterable[str] = ('v1', 'v2')) -> dict[str, RefinementResult]:
        """Compute common projection/alignment once when requesting both."""
        versions = tuple(dict.fromkeys(v.lower() for v in versions))
        if not versions or not set(versions) <= {'v1', 'v2'}:
            raise ValueError('versions must be v1 and/or v2')
        original = np.asarray(stereo)
        prior = np.asarray(mono, dtype=np.float32)
        cal, cfg = self.calibration, self.config
        if original.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('stereo must be float32/float64 floating-point disparity in px; decode Q6 first')
        if original.shape != (cal.depth.height, cal.depth.width):
            raise ValueError('stereo shape differs from depth calibration')
        if prior.shape != (cal.rgb.height, cal.rgb.width) or not np.isfinite(prior).all():
            raise ValueError('mono must be finite and match native RGB calibration shape')
        if min(original.shape+prior.shape) < 2:
            raise ValueError('images must be at least 2x2')
        projection = project_depth_disparity_to_rgb(original, cal)
        evidence = align_windows(projection.equivalent_disparity, prior, cfg.alignment,
                                 cfg.threshold_px if 'v2' in versions else None)
        v1, removed_v1 = reject_from_rgb(original, projection.source_flat_index,
                                         evidence['residual_px'], evidence['edge_band'], cfg.threshold_px)
        results = {}
        if 'v1' in versions:
            results['v1'] = RefinementResult('v1', v1, removed_v1, np.zeros_like(removed_v1),
                                             v1.copy(), dict(alignment=evidence['report']))
        if 'v2' in versions:
            rgb_removed, _ = consensus_mask(evidence, cfg)
            removed = source_mask(original, projection.source_flat_index, rgb_removed)
            if (removed & ~removed_v1).any():
                raise RuntimeError('V2 consensus must only veto V1 deletions')
            rejected = original.copy()
            rejected[removed] = np.nan
            guide, safe = mapped_guide(original, projection.source_flat_index, evidence, removed_v1, cfg)
            smoothed, report = bounded_guided_smooth(rejected, guide, safe, cfg)
            interior_report = None
            if cfg.interior.enabled:
                protected = source_mask(original, projection.source_flat_index,
                    dilate(evidence['mono_edge'], cfg.boundary_guard_radius)) | removed_v1
                smoothed, interior_report = multiscale_interior_smooth(
                    original, smoothed, guide, safe, cfg, protected=protected)
            changed = valid_disparity(smoothed) & (np.abs(smoothed-original) > .05)
            results['v2'] = RefinementResult('v2', smoothed, removed, changed, rejected,
                                             dict(alignment=evidence['report'], smoothing=report,
                                                  v1_removed=int(removed_v1.sum())))
            if interior_report is not None:
                results['v2'].diagnostics['interior_smoothing'] = interior_report
        for result in results.values():
            if (valid_disparity(result.disparity) & ~valid_disparity(original)).any():
                raise RuntimeError('refinement filled original invalid pixels')
            if not np.array_equal(valid_disparity(result.disparity),
                                  valid_disparity(result.rejected_disparity)):
                raise RuntimeError('smoothing changed the rejected validity mask')
        return results
