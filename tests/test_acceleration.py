"""Native execution preserves robust fits and deterministic overlap reduction."""
import os
import subprocess
import sys

import numpy as np
import pytest

from disp_fusion import InteriorSmoothConfig, RefinementConfig
from disp_fusion import _acceleration
from disp_fusion.alignment import robust_affine
from disp_fusion.smoothing import _robust_local_plane, _sample_indices, multiscale_interior_smooth
from test_interior_smoothing import ramp_example


def reference_irls(design, values, iterations, floor, epsilon):
    weights = np.ones(len(values))
    for _ in range(iterations):
        root = np.sqrt(weights)
        beta, _, rank, _ = np.linalg.lstsq(design*root[:, None], values*root, rcond=None)
        residual = values-design@beta
        scale = max(floor, 1.4826*np.median(abs(residual-np.median(residual))))
        weights = np.minimum(1., 1.5*scale/np.maximum(abs(residual), epsilon))
    return beta, residual, rank


@pytest.mark.parametrize('count', [32, 128, 512])
def test_native_plane_keeps_svd_irls_result(count):
    rng = np.random.default_rng(count)
    design = np.column_stack((rng.uniform(-1, 1, (count, 2)), np.ones(count)))
    values = design@np.array([.03, -.04, 20.])+rng.normal(0, .04, count)
    values[::11] += 3
    expected, _, _ = reference_irls(design, values, 5, .02, 1e-12)
    actual = _robust_local_plane(design, values, 5, .02)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


def test_native_plane_still_rejects_rank_deficiency():
    assert _robust_local_plane(np.ones((128, 3)), np.ones(128), 5, .02) is None


def test_native_affine_keeps_original_fit_and_gate_statistic():
    rng = np.random.default_rng(42)
    mono = rng.uniform(0, 1, 1200).astype(np.float32)
    stereo = (20+30*mono+rng.normal(0, .1, 1200)).astype(np.float32)
    stereo[::17] += 10
    design = np.column_stack((mono.astype(float), np.ones(len(mono))))
    beta, residual, _ = reference_irls(design, stereo.astype(float), 8, .15, 1e-8)
    actual, gate = robust_affine(mono, stereo)
    np.testing.assert_allclose(actual, beta, rtol=0, atol=2e-12)
    assert gate == pytest.approx(np.percentile(abs(residual), 80), abs=2e-12)


def test_sampling_matches_numpy_linspace_integer_rounding():
    for count in [16, 17, 127, 128, 511, 512, 513, 1023, 1024, 3969, 16129]:
        indices = np.arange(count, dtype=np.int64)*3
        for maximum in [16, 128, 512]:
            expected = indices[np.linspace(0, count-1, min(count, maximum), dtype=int)]
            assert np.array_equal(_sample_indices(indices, maximum), expected)


@pytest.mark.parametrize('percentile', [80, 90])
def test_percentile_preserves_numpy_interpolation_at_strict_gates(percentile):
    rng = np.random.default_rng(410)
    for count in [16, 17, 128, 129, 512, 1200]:
        values = rng.uniform(1e-6, 5, count)
        expected = float(np.percentile(values, percentile))
        assert _acceleration.linear_percentile(values, percentile) == expected


def test_parallel_fit_order_does_not_change_pixels_or_window_gates(monkeypatch):
    _, original, guide, safe = ramp_example()
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    monkeypatch.setenv('DISP_FUSION_WORKERS', '1')
    sequential, first = multiscale_interior_smooth(original, original, guide, safe, cfg)
    monkeypatch.setenv('DISP_FUSION_WORKERS', '4')
    parallel, second = multiscale_interior_smooth(original, original, guide, safe, cfg)
    assert np.array_equal(sequential, parallel, equal_nan=True)
    assert {k:v for k,v in first.items() if k != 'seconds'} == {k:v for k,v in second.items() if k != 'seconds'}


def test_acceleration_can_be_disabled_before_import():
    env = dict(os.environ, DISP_FUSION_ACCELERATION='0', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-c',
        'from disp_fusion._acceleration import ENABLED, window_workers; '
        'from disp_fusion.smoothing import _robust_local_plane; '
        'assert not ENABLED and window_workers() == 1; '
        'assert not hasattr(_robust_local_plane, "py_func")'], env=env, check=True)


def test_worker_configuration_is_checked(monkeypatch):
    monkeypatch.setattr(_acceleration, 'ENABLED', True)
    monkeypatch.setenv('DISP_FUSION_WORKERS', '0')
    with pytest.raises(ValueError, match='positive integer'):
        _acceleration.window_workers()


def test_read_only_cache_does_not_prevent_import_or_compilation(monkeypatch):
    calls = []
    def fake_njit(**kwargs):
        calls.append(kwargs)
        def decorate(function):
            if kwargs['cache']:
                raise RuntimeError('cannot cache function: no locator available')
            return function
        return decorate
    monkeypatch.setattr(_acceleration, 'ENABLED', True)
    monkeypatch.setattr(_acceleration, 'njit', fake_njit, raising=False)
    def example():
        return 5
    assert _acceleration.compiled(example)() == 5
    assert [call['cache'] for call in calls] == [True, False]
    assert all(not call['fastmath'] and call['nogil'] for call in calls)
