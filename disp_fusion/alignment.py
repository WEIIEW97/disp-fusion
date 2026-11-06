"""Reject-only local mono/stereo comparison. No depth synthesis or matching."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


from .config import WindowConfig
from ._acceleration import compiled, linear_percentile


def valid_disparity(values):
    return np.isfinite(values) & (values > .05)


def dilate(mask, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


@compiled
def robust_affine(mono, stereo, iterations=8):
    """Huber IRLS, positive slope; return None rather than invent alignment."""
    x = np.asarray(mono, np.float64)
    y = np.asarray(stereo, np.float64)
    design = np.column_stack((x, np.ones(len(x))))
    weight = np.ones(len(x))
    for _ in range(iterations):
        beta = np.linalg.lstsq(design*np.sqrt(weight[:, None]), y*np.sqrt(weight),
                              rcond=np.finfo(np.float64).eps*max(design.shape))[0]
        residual = y-design@beta
        sigma = max(.15, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weight = np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1e-8))
    if not np.isfinite(beta).all() or beta[0] <= 0:
        return None
    return beta, float(linear_percentile(np.abs(residual), 80))


def align_windows(stereo_rgb, mono, config=WindowConfig(), vote_threshold_px=None):
    """Both fields are RGB-frame. Stereo units are fB/Z_rgb equivalent px.

    Nearest fill/median are used ONLY to locate edges and choose anchors.
    Overlapping accepted windows average their mono predictions; stereo values
    never get replaced by those predictions. No previous fusion is an input.
    """
    if stereo_rgb.shape != mono.shape or stereo_rgb.ndim != 2:
        raise ValueError('expected same-size 2-D RGB-frame arrays')
    if config.window < 5 or config.window % 2 != 1 or config.stride < 1:
        raise ValueError('window must be odd >=5 and stride positive')
    if vote_threshold_px is not None and (not np.isfinite(vote_threshold_px) or vote_threshold_px <= 0):
        raise ValueError('vote threshold must be positive and finite')
    valid = valid_disparity(stereo_rgb) & np.isfinite(mono)
    shape = stereo_rgb.shape
    empty = np.full(shape, np.nan, np.float32)
    if not valid.any() or not np.isfinite(mono).any():
        return dict(aligned_mono=empty, residual_px=empty.copy(),
                    edge_band=np.zeros(shape, bool), anchors=np.zeros(shape, bool),
                    mono_normalized=np.zeros(shape, np.float32), mono_edge=np.zeros(shape, bool),
                    vote_count=np.zeros(shape, np.uint16), positive_votes=np.zeros(shape, np.uint16),
                    negative_votes=np.zeros(shape, np.uint16),
                    fit_coverage=np.zeros(shape, bool), report=dict(accepted_windows=0, attempted_windows=0))
    lo, hi = np.percentile(mono[np.isfinite(mono)], [2, 98])
    m = np.clip((mono-lo)/max(float(hi-lo), 1e-6), 0, 1).astype(np.float32)
    m[~np.isfinite(m)] = 0
    smoothed = cv2.GaussianBlur(m, (0, 0), 1.)
    my, mx = np.gradient(smoothed)
    mono_edge = np.hypot(mx, my) > config.mono_gradient
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = stereo_rgb[tuple(nearest)].astype(np.float32)
    median = cv2.medianBlur(filled, 5)
    sy, sx = np.gradient(median)
    stereo_edge = np.hypot(sx, sy) > config.stereo_gradient_px
    edge_band = dilate(mono_edge | stereo_edge, config.edge_radius)
    anchors = (valid & ~dilate(mono_edge | stereo_edge, 3)
               & (np.abs(stereo_rgb-median) <= config.anchor_deviation_px))
    total = np.zeros(shape, np.float64)
    weights = np.zeros(shape, np.float64)
    vote_count = np.zeros(shape, np.uint16)
    positive_votes = np.zeros(shape, np.uint16)
    negative_votes = np.zeros(shape, np.uint16)
    radius = config.window//2
    accepted = attempted = 0
    h, w = shape
    for cy in range(0, h, config.stride):
        for cx in range(0, w, config.stride):
            y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
            x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
            sl = np.s_[y0:y1, x0:x1]
            if not (edge_band[sl] & valid[sl]).any():
                continue
            attempted += 1
            yy, xx = np.where(anchors[sl])
            if len(xx) < config.minimum_anchors:
                continue
            take = np.linspace(0, len(xx)-1, min(1200, len(xx)), dtype=int)
            yy, xx = yy[take]+y0, xx[take]+x0
            values = m[yy, xx]
            if np.percentile(values, 95)-np.percentile(values, 5) < config.minimum_mono_span:
                continue
            fit = robust_affine(values, stereo_rgb[yy, xx])
            if fit is None or fit[1] > config.maximum_fit_p80_px:
                continue
            beta, _ = fit
            wy = np.maximum(.05, 1-np.abs(np.arange(y0, y1)-cy)/(radius+1))
            wx = np.maximum(.05, 1-np.abs(np.arange(x0, x1)-cx)/(radius+1))
            weight = wy[:, None]*wx[None, :]
            prediction = beta[0]*m[sl]+beta[1]
            total[sl] += weight*prediction
            weights[sl] += weight
            if vote_threshold_px is not None:
                measured = valid[sl]
                difference = stereo_rgb[sl]-prediction
                vote_count[sl] += measured
                positive_votes[sl] += measured & (difference > vote_threshold_px)
                negative_votes[sl] += measured & (difference < -vote_threshold_px)
            accepted += 1
    aligned = np.full(shape, np.nan, np.float32)
    covered = (weights > 0) & np.isfinite(mono)
    aligned[covered] = (total[covered]/weights[covered]).astype(np.float32)
    residual = np.full(shape, np.nan, np.float32)
    measured = valid & covered
    residual[measured] = np.abs(stereo_rgb[measured]-aligned[measured])
    return dict(aligned_mono=aligned, residual_px=residual, edge_band=edge_band,
                mono_normalized=m, mono_edge=mono_edge, vote_count=vote_count,
                positive_votes=positive_votes, negative_votes=negative_votes,
                anchors=anchors, fit_coverage=covered,
                report=dict(accepted_windows=accepted, attempted_windows=attempted,
                            anchor_pixels=int(anchors.sum()), edge_pixels=int((edge_band & valid).sum()),
                            compared_edge_pixels=int((edge_band & measured).sum())))


def reject_from_rgb(original, source_flat_index, residual_px, edge_band, threshold_px):
    """Map z-buffer winners back to source. Unseen pixels are left untouched."""
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError('threshold must be positive and finite')
    if not (source_flat_index.shape == residual_px.shape == edge_band.shape):
        raise ValueError('RGB index, residual and band shapes must agree')
    selected = ((source_flat_index >= 0) & edge_band
                & np.isfinite(residual_px) & (residual_px > threshold_px))
    source = source_flat_index[selected].astype(np.int64)
    if np.any(source >= original.size):
        raise ValueError('source index exceeds original image')
    removed = np.zeros(original.size, bool)
    removed[source] = True
    removed = removed.reshape(original.shape) & valid_disparity(original)
    output = original.copy()
    output[removed] = np.nan
    kept = valid_disparity(output)
    assert not (kept & ~valid_disparity(original)).any()
    assert np.array_equal(output[kept], original[kept])
    return output, removed
