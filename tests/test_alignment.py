import numpy as np

from disp_fusion.config import WindowConfig
from disp_fusion.alignment import (
    align_windows, reject_from_rgb, robust_affine,
)


def test_affine_fit_resists_sparse_outliers():
    x = np.linspace(0, 1, 400)
    y = 30*x+10
    y[::20] += 25
    fit = robust_affine(x, y)
    assert fit is not None
    assert np.allclose(fit[0], [30, 10], atol=.1)


def test_rejection_only_removes_selected_valid_source_pixels():
    original = np.array([[10., np.nan, 30.], [0., 50., 60.]], np.float32)
    sources = np.array([[0, 1, 2], [3, 4, -1]])
    residual = np.array([[8., 8., 1.], [8., 8., 8.]])
    band = np.array([[True, True, True], [True, False, True]])
    output, removed = reject_from_rgb(original, sources, residual, band, 4)
    assert removed.sum() == 1 and removed[0, 0]
    assert np.isnan(output[0, 0]) and np.isnan(output[0, 1])
    assert output[1, 0] == 0 and output[1, 2] == 60
    valid = np.isfinite(output) & (output > .05)
    assert np.array_equal(output[valid], original[valid])


def test_no_valid_stereo_is_noop():
    stereo = np.full((20, 30), np.nan, np.float32)
    evidence = align_windows(stereo, np.ones_like(stereo))
    assert not evidence['fit_coverage'].any()
    assert evidence['report']['accepted_windows'] == 0


def test_constant_mono_cannot_invent_local_alignment():
    y, x = np.indices((80, 100))
    stereo = np.where(x < 50, 20., 60.).astype(np.float32)
    evidence = align_windows(stereo, np.ones_like(stereo))
    assert not evidence['fit_coverage'].any()


def test_clean_step_preserved_but_sparse_edge_spikes_detected():
    y, x = np.indices((128, 160))
    mono = (x >= 80).astype(np.float32)+x*.0005
    stereo = (20+40*mono).astype(np.float32)
    stereo[30:100:4, 78] += 12
    cfg = WindowConfig(window=65, stride=24)
    e = align_windows(stereo, mono, cfg)
    sources = np.arange(stereo.size).reshape(stereo.shape)
    output, removed = reject_from_rgb(stereo, sources, e['residual_px'], e['edge_band'], 4)
    assert removed[30:100:4, 78].all()
    assert removed.sum() == len(range(30, 100, 4))
    kept = np.isfinite(output)
    assert np.array_equal(output[kept], stereo[kept])


def test_thresholds_have_nested_deletion_masks():
    original = np.ones((3, 4), np.float32)*20
    index = np.arange(12).reshape(3, 4)
    residual = index.astype(float)
    band = np.ones((3, 4), bool)
    _, low = reject_from_rgb(original, index, residual, band, 2)
    _, high = reject_from_rgb(original, index, residual, band, 8)
    assert not (high & ~low).any()
