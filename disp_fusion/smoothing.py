"""Bounded mono-guided graph smoothing shared by the refinement pipeline."""
import numpy as np
import cv2
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from scipy.ndimage import distance_transform_edt
from .alignment import valid_disparity, dilate
from .config import RefinementConfig
from ._acceleration import ENABLED as NATIVE_ENABLED, compiled, window_workers, linear_percentile


def bounded_guided_smooth(rejected, guide, safe, cfg=RefinementConfig()):
    """Jacobi steps for data-anchored disparity graph smoothing.

    E(d)=1/2 sum_i(d_i-s_i)^2 + alpha/2 sum_(i,j)w_ij(d_i-d_j)^2.
    Weights are fixed from original retained stereo and mono. Each update
    is clipped to s_i +/- max_change_px. Invalid and unsafe nodes never move.
    """
    if not (rejected.shape == guide.shape == safe.shape) or rejected.ndim != 2:
        raise ValueError('expected same-size 2-D arrays')
    if (cfg.smooth_iterations < 0 or cfg.smooth_strength < 0 or cfg.max_change_px < 0
            or cfg.mono_sigma <= 0 or cfg.stereo_sigma_px <= 0):
        raise ValueError('invalid smoothing parameters')
    valid = valid_disparity(rejected)
    safe = safe & valid & np.isfinite(guide)
    if cfg.smooth_iterations == 0 or cfg.smooth_strength == 0 or cfg.max_change_px == 0:
        return rejected.copy(), dict(safe_pixels=int(safe.sum()), movable_pixels=0,
            changed_over_005px=0, maximum_change_px=0.)
    data = np.where(valid, rejected, 0).astype(np.float32)
    degree = np.zeros_like(data)
    edges = []
    # Undirected four-neighbor graph; slicing prevents image-boundary wrap.
    for a, b in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        pair = safe[a] & safe[b] & (np.abs(data[a]-data[b]) <= cfg.stereo_jump_limit_px)
        md = np.where(pair, guide[a]-guide[b], 0)
        sd = data[a]-data[b]
        weight = np.where(pair, np.exp(-.5*((md/cfg.mono_sigma)**2
                                           +(sd/cfg.stereo_sigma_px)**2)), 0).astype(np.float32)
        degree[a] += weight; degree[b] += weight
        edges.append((a, b, weight))
    current = data.copy()
    movable = safe & (degree > 0)
    for _ in range(cfg.smooth_iterations):
        neighbors = np.zeros_like(data)
        for a, b, weight in edges:
            neighbors[a] += weight*current[b]
            neighbors[b] += weight*current[a]
        proposal = (data+cfg.smooth_strength*neighbors)/(1+cfg.smooth_strength*degree)
        current = np.where(movable, np.clip(proposal, data-cfg.max_change_px,
                                           data+cfg.max_change_px), data).astype(np.float32)
    result = rejected.copy()
    result[movable] = current[movable]
    assert np.array_equal(valid_disparity(result), valid)
    assert np.all(np.abs(result[valid]-rejected[valid]) <= cfg.max_change_px+1e-4)
    return result, dict(safe_pixels=int(safe.sum()), movable_pixels=int(movable.sum()),
                        changed_over_005px=int((valid & (np.abs(result-data) > .05)).sum()),
                        maximum_change_px=float(np.max(np.abs(result[valid]-rejected[valid]))) if valid.any() else 0.)


@compiled
def _robust_local_plane(design, values, iterations, scale_floor):
    """Small, normalized-coordinate IRLS fit; fail rather than invent a slope."""
    weights = np.ones(len(values), np.float64)
    for _ in range(iterations):
        root = np.sqrt(weights)
        beta, _, rank, _ = np.linalg.lstsq(design*root[:, None], values*root,
                                          rcond=np.finfo(np.float64).eps*max(design.shape))
        if rank < 3 or not np.isfinite(beta).all():
            return None
        residual = values-design@beta
        scale = max(scale_floor, 1.4826*np.median(np.abs(residual-np.median(residual)))) # 1.4826 is the Gaussian consistency factor for the median absolute deviation (MAD)
        weights = np.minimum(1., 1.5*scale/np.maximum(np.abs(residual), 1e-12))
    return beta


@compiled
def _interior_model(design, stereo, mono, baseline, train, check, iterations, stereo_limit, mono_limit):
    """The unchanged fit, held-out gates and median correction, in one call."""
    beta = _robust_local_plane(design[train], stereo[train], iterations, .02)
    mono_beta = _robust_local_plane(design[train], mono[train], iterations, 1e-5)
    if beta is None or mono_beta is None:
        return None
    if linear_percentile(np.abs(stereo[check]-design[check]@beta), 90) > stereo_limit:
        return None
    if linear_percentile(np.abs(mono[check]-design[check]@mono_beta), 90) > mono_limit:
        return None
    offset = np.median(design[:, 0]*beta[0]+design[:, 1]*beta[1]+beta[2]-baseline)
    return beta, offset


@compiled
def _sample_indices(indices, maximum):
    """Match np.linspace(..., dtype=int), including its explicit endpoint."""
    count = min(len(indices), maximum)
    take = np.zeros(count, np.int64)
    if count > 1:
        step = (len(indices)-1)/(count-1)
        for i in range(count-1):
            take[i] = int(i*step)
        take[-1] = len(indices)-1
    return indices[take]


@compiled
def _evaluate_window(ay, ax, y0, x0, cy, cx, radius, original, baseline, guide,
                     maximum_samples, iterations, stereo_limit, mono_limit):
    holdout = ((ay+y0)*3+(ax+x0)*5) % 7 == 0
    train, check = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    if min(len(train), len(check)) < 16:
        return None
    train = _sample_indices(train, maximum_samples)
    check = _sample_indices(check, maximum_samples)
    design = np.column_stack(((ax+x0-cx)/radius, (ay+y0-cy)/radius, np.ones(len(ax))))
    if NATIVE_ENABLED:
        stereo_values = np.empty(len(ax), np.float64)
        mono_values = np.empty(len(ax), np.float64)
        baseline_values = np.empty(len(ax), np.float64)
        for i in range(len(ax)):
            stereo_values[i] = original[ay[i]+y0, ax[i]+x0]
            mono_values[i] = guide[ay[i]+y0, ax[i]+x0]
            baseline_values[i] = baseline[ay[i]+y0, ax[i]+x0]
    else:
        stereo_values = original[ay+y0, ax+x0].astype(np.float64)
        mono_values = guide[ay+y0, ax+x0].astype(np.float64)
        baseline_values = baseline[ay+y0, ax+x0].astype(np.float64)
    return _interior_model(design, stereo_values, mono_values, baseline_values, train, check,
                           iterations, stereo_limit, mono_limit)


def _evaluate_interior_job(job, *, original, baseline, guide, config):
    size, cy, cx, y0, y1, x0, x1, component, ay, ax = job
    return _evaluate_window(ay, ax, y0, x0, cy, cx, size//2, original, baseline, guide,
        config.maximum_samples, config.fit_iterations, config.stereo_fit_p90_px, config.mono_fit_p90)


@compiled
def _accumulate_window(correction_sum, weight_sum, baseline, eligible, component,
                       beta, offset, y0, y1, x0, x1, cy, cx, radius, scale):
    """Serial IEEE arithmetic in the original window order; no fast-math."""
    for y in range(y0, y1):
        wy = max(0., 1.-abs(y-cy)/radius)
        for x in range(x0, x1):
            if not eligible[y, x] or not component[y-y0, x-x0]:
                continue
            wx = max(0., 1.-abs(x-cx)/radius)
            weight = wy*wx*scale
            prediction = beta[0]*(x-cx)/radius+beta[1]*(y-cy)/radius+beta[2]
            delta = prediction-baseline[y, x]-offset
            correction_sum[y, x] += weight*delta
            weight_sum[y, x] += weight


def _interior_jobs(valid, anchors, integral, config):
    """Prepare topology once; every worker reads the same immutable inputs."""
    h, w = valid.shape
    jobs = []
    for size in config.windows:
        radius = size//2
        stride = max(1, round(size*config.stride_fraction))
        for cy in range(0, h, stride):
            for cx in range(0, w, stride):
                y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
                x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
                if min(y1-y0, x1-x0) < 7:
                    continue
                if integral[y1, x1]-integral[y0, x1]-integral[y1, x0]+integral[y0, x0]:
                    continue
                component = valid[y0:y1, x0:x1]
                if not component[cy-y0, cx-x0]:
                    continue
                if not component.all():
                    _, labels = cv2.connectedComponents(component.astype(np.uint8), connectivity=4)
                    component = labels == labels[cy-y0, cx-x0]
                ay, ax = np.where(anchors[y0:y1, x0:x1] & component)
                if len(ax) < config.minimum_anchors or len(ax)/(y1-y0)/(x1-x0) < config.minimum_coverage:
                    continue
                jobs.append((size, cy, cx, y0, y1, x0, x1, component, ay, ax))
    return jobs


def multiscale_interior_smooth(original, baseline, guide, safe, cfg=RefinementConfig(), *, protected=None):
    """Refine V2 interiors while keeping its validity and boundaries fixed.

    Stereo determines absolute affine-disparity trends; mono only gates local
    planarity. Model windows avoid supplied boundaries and stereo jumps. Small
    invalid holes are excluded; locally disconnected valid components cannot
    share a model. Original-valid samples in tiny guidance sampling gaps may
    be modified only when surrounded by an accepted interior model.
    The final bound is relative to original stereo, not an additional allowance.
    """
    started = perf_counter()
    original, baseline, guide = map(np.asarray, (original, baseline, guide))
    safe = np.asarray(safe, dtype=bool)
    if not (original.shape == baseline.shape == guide.shape == safe.shape) or original.ndim != 2:
        raise ValueError('interior smoothing requires same-size 2-D arrays')
    if original.dtype.kind != 'f' or baseline.dtype.kind != 'f':
        raise ValueError('interior smoothing requires floating-point disparity')
    valid = valid_disparity(baseline)
    if (valid & ~valid_disparity(original)).any():
        raise ValueError('baseline filled originally invalid samples')
    initial_change = float(np.abs(baseline[valid]-original[valid]).max(initial=0))
    if initial_change > cfg.max_change_px+1e-4:
        raise ValueError('baseline already exceeds the total original-disparity bound')
    config = cfg.interior
    report = dict(enabled=config.enabled, attempted_windows=0, accepted_windows=0,
                  accepted_by_scale={str(w):0 for w in config.windows}, modeled_pixels=0,
                  changed_over_005px=0, maximum_additional_change_px=0., maximum_total_change_px=initial_change,
                  guidance_gap_targets=0)
    result = baseline.copy()
    if not config.enabled or config.strength == 0 or cfg.max_change_px == 0 or not valid.any():
        report['seconds'] = perf_counter()-started
        return result, report
    if protected is None:
        protected = ~safe
    protected = np.asarray(protected, dtype=bool)
    if protected.shape != original.shape:
        raise ValueError('interior protection mask shape mismatch')
    jump = np.zeros(original.shape, bool)
    for a, b in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        step = valid[a] & valid[b] & (np.abs(original[a]-original[b]) > cfg.stereo_jump_limit_px)
        jump[a] |= step; jump[b] |= step
    structural_barrier = protected | dilate(jump, config.boundary_guard_radius)
    barrier = ~valid | structural_barrier
    available = safe & valid & np.isfinite(guide) & ~barrier
    guidance_gaps = np.zeros_like(available)
    if available.any() and config.guide_hole_radius > 0:
        nearby = distance_transform_edt(~available) <= config.guide_hole_radius
        guidance_gaps = valid & ~np.isfinite(guide) & ~barrier & nearby
    eligible = available | guidance_gaps
    report.update(eligible_pixels=int(eligible.sum()), frozen_pixels=int((valid & ~eligible).sum()),
                  frozen_changed_pixels=0, protected_changed_pixels=0)
    if not eligible.any():
        report['seconds'] = perf_counter()-started
        return result, report
    # Median screening is auxiliary; invalid samples are never model anchors
    # or output targets, and disconnected local components cannot share a fit.
    median = cv2.medianBlur(np.where(valid_disparity(original), original, 0).astype(np.float32), 5)
    anchors = available & (np.abs(original-median) <= config.anchor_deviation_px)
    integral = cv2.integral(structural_barrier.astype(np.uint8), sdepth=cv2.CV_32S)
    correction_sum = np.zeros(original.shape, np.float64)
    weight_sum = np.zeros(original.shape, np.float64)
    jobs = _interior_jobs(valid, anchors, integral, config)
    report['attempted_windows'] = len(jobs)
    evaluate = partial(_evaluate_interior_job, original=original, baseline=baseline, guide=guide, config=config)
    workers = window_workers()
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 and jobs else None
    try:
        models = executor.map(evaluate, jobs) if executor else map(evaluate, jobs)
        # map preserves window order: workers NEVER accumulate overlapping data.
        for job, model in zip(jobs, models):
            if model is None:
                continue
            size, cy, cx, y0, y1, x0, x1, component, _, _ = job
            radius = size//2
            beta, offset = model
            sl = np.s_[y0:y1, x0:x1]
            if NATIVE_ENABLED:
                _accumulate_window(correction_sum, weight_sum, baseline, eligible, component,
                    beta, offset, y0, y1, x0, x1, cy, cx, radius, (size/config.windows[0])**2)
            else:
                yy, xx = np.indices((y1-y0, x1-x0), dtype=np.float64)
                prediction = beta[0]*(xx+x0-cx)/radius+beta[1]*(yy+y0-cy)/radius+beta[2]
                delta = prediction-baseline[sl]-offset
                wy = np.maximum(0., 1.-np.abs(np.arange(y0, y1)-cy)/radius)
                wx = np.maximum(0., 1.-np.abs(np.arange(x0, x1)-cx)/radius)
                weights = wy[:, None]*wx[None, :]*(size/config.windows[0])**2
                weights *= eligible[sl] & component
                correction_sum[sl] += weights*delta
                weight_sum[sl] += weights
            report['accepted_windows'] += 1
            report['accepted_by_scale'][str(size)] += 1
    finally:
        if executor:
            executor.shutdown(wait=True)
    modeled = eligible & (weight_sum > 0)
    # A lone vanishing-weight window must fade into the unchanged baseline.
    # Normalizing by arbitrarily small support would cancel the edge taper.
    proposal = baseline[modeled]+config.strength*correction_sum[modeled]/np.maximum(weight_sum[modeled], 1.)
    proposal = np.clip(proposal, original[modeled]-cfg.max_change_px, original[modeled]+cfg.max_change_px)
    proposal = np.where(proposal > .05, proposal, baseline[modeled])
    result[modeled] = proposal
    assert np.array_equal(valid_disparity(result), valid)
    assert np.array_equal(result[~eligible], baseline[~eligible], equal_nan=True)
    assert np.abs(result[valid]-original[valid]).max(initial=0) <= cfg.max_change_px+1e-4
    report.update(modeled_pixels=int(modeled.sum()),
        guidance_gap_targets=int((modeled & guidance_gaps).sum()),
        frozen_changed_pixels=int((valid & ~eligible & (result != baseline)).sum()),
        protected_changed_pixels=int((valid & protected & (result != baseline)).sum()),
        changed_over_005px=int((np.abs(result[valid]-baseline[valid]) > .05).sum()),
        maximum_additional_change_px=float(np.abs(result[valid]-baseline[valid]).max(initial=0)),
        maximum_total_change_px=float(np.abs(result[valid]-original[valid]).max(initial=0)),
        seconds=perf_counter()-started)
    return result, report
