"""Optional interior smoothing: stripes, ramps, masks, curvature, and total cap."""
from dataclasses import replace
import json

import numpy as np
import pytest

from disp_fusion import DepthRefiner, InteriorSmoothConfig, RefinementConfig
from disp_fusion.cli import load_config
from disp_fusion.smoothing import bounded_guided_smooth, multiscale_interior_smooth
from test_api import calibration, example


def ramp_example():
    y, x = np.indices((128, 160))
    plane = 30.+.02*x+.01*y
    rng = np.random.default_rng(731)
    original = (plane+.22*np.sin(2*np.pi*x/24)+rng.normal(0, .04, plane.shape)).astype(np.float32)
    guide = (.2+.0001*x+.0002*y).astype(np.float32)
    safe = np.ones(original.shape, bool)
    return plane, original, guide, safe


def test_stripes_are_reduced_while_global_ramp_is_preserved():
    plane, original, guide, safe = ramp_example()
    baseline, _ = bounded_guided_smooth(original, guide, safe)
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    output, report = multiscale_interior_smooth(original, baseline, guide, safe, cfg)
    core = np.s_[32:-32, 32:-32]
    assert np.std((output-plane)[core]) < .55*np.std((baseline-plane)[core])
    y, x = np.indices(original.shape)
    design = np.c_[x[core].ravel(), y[core].ravel(), np.ones(x[core].size)]
    beta = np.linalg.lstsq(design, output[core].ravel(), rcond=None)[0]
    np.testing.assert_allclose(beta[:2], [.02, .01], atol=.0007)
    assert abs(np.median((output-original)[core])) < .04
    assert report['accepted_by_scale']['127'] > 0
    assert np.abs(output-original).max() <= .75001


def test_invalid_new_holes_and_protected_samples_are_bit_exact():
    _, original, guide, safe = ramp_example()
    original[20:25, 30:35] = np.nan
    baseline = original.copy(); baseline[50:55, 80:85] = np.nan
    protected = np.zeros(original.shape, bool); protected[:, 76:80] = True
    safe &= ~protected; safe[70:75, 20:25] = False
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    output, report = multiscale_interior_smooth(original, baseline, guide, safe, cfg, protected=protected)
    assert np.array_equal(np.isfinite(output), np.isfinite(baseline))
    frozen = ~safe | ~np.isfinite(baseline)
    assert np.array_equal(output[frozen], baseline[frozen], equal_nan=True)
    assert report['modeled_pixels'] > 0


def test_strong_stereo_and_mono_step_cannot_leak_across_surfaces():
    _, original, guide, safe = ramp_example()
    original[:, 80:] += 20
    guide[:, 80:] += .5
    protected = np.zeros(original.shape, bool); protected[:, 77:83] = True
    safe &= ~protected
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    output, _ = multiscale_interior_smooth(original, original, guide, safe, cfg, protected=protected)
    assert np.array_equal(output[:, 77:83], original[:, 77:83])
    assert np.abs(output-original).max() <= .75001


def test_curved_mono_rejects_a_large_planar_model():
    y, x = np.indices((96, 96))
    guide = (((x-48)/48)**2+((y-48)/48)**2).astype(np.float32)
    original = (30+.01*x+.1*np.sin(x)).astype(np.float32)
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True, windows=(63,)))
    output, report = multiscale_interior_smooth(original, original, guide, np.ones_like(original, bool), cfg)
    assert report['attempted_windows'] > 0
    assert report['accepted_windows'] == 0
    assert np.array_equal(output, original)


def test_locally_disconnected_regions_do_not_share_an_affine_model():
    rng = np.random.default_rng(13)
    original = (30+rng.normal(0, .04, (96, 160))).astype(np.float32)
    original[:, 81:] += .6  # Below the stereo-jump guard: the hole must separate models.
    original[:, 79:81] = np.nan
    guide = np.full_like(original, .2)
    safe = np.isfinite(original)
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    output, report = multiscale_interior_smooth(original, original, guide, safe, cfg,
                                               protected=np.zeros_like(safe))
    assert np.isnan(output[:, 79:81]).all()
    assert abs(np.nanmedian(output[:, :79])-30) < .02
    assert abs(np.nanmedian(output[:, 81:])-30.6) < .02
    assert report['accepted_windows'] > 0


def test_second_stage_uses_total_original_bound_not_an_additional_allowance():
    _, original, guide, safe = ramp_example()
    baseline = original+.6
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    output, _ = multiscale_interior_smooth(original, baseline, guide, safe, cfg)
    assert np.abs(output-original).max() <= .75001


def test_disabled_stage_is_bit_exact_for_float64():
    _, original, guide, safe = ramp_example()
    baseline = original.astype(np.float64)+.01
    output, report = multiscale_interior_smooth(original, baseline, guide, safe)
    assert np.array_equal(output, baseline) and output.dtype == np.float64
    assert not report['enabled']
    assert report['maximum_total_change_px'] == pytest.approx(.01, abs=1e-7)


def test_tiny_guidance_gaps_are_not_confused_with_invalid_stereo_holes():
    _, original, guide, safe = ramp_example()
    original[20:24, 40:44] = np.nan
    guide[:, 30] = np.nan; safe[:, 30] = False
    guide[:, 70:90] = np.nan; safe[:, 70:90] = False
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    output, report = multiscale_interior_smooth(original, original, guide, safe, cfg,
                                               protected=np.zeros_like(safe))
    assert np.isnan(output[20:24, 40:44]).all()
    assert np.array_equal(output[:, 77:83], original[:, 77:83])
    assert report['guidance_gap_targets'] > 0
    assert np.any(np.abs(output[:, 30]-original[:, 30]) > .01)


def test_model_coverage_edges_fade_into_unchanged_baseline():
    y, x = np.indices((96, 96))
    original = (30+.2*np.cos(2*np.pi*x/12)).astype(np.float32)
    guide = np.full_like(original, .2)
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True, windows=(31,), stride_fraction=1.))
    output, report = multiscale_interior_smooth(original, original, guide, np.ones_like(original, bool), cfg)
    assert report['accepted_windows'] > 0
    assert np.array_equal(output[:, 15], original[:, 15])
    assert np.max(abs(output[:, 14]-original[:, 14])) < .04


def test_pipeline_preserves_v1_v2_masks_and_frozen_boundary():
    original, mono = example()
    before = DepthRefiner(calibration()).refine_many(original, mono)
    after = DepthRefiner(calibration(), RefinementConfig(interior=InteriorSmoothConfig(enabled=True))).refine_many(original, mono)
    assert np.array_equal(before['v1'].disparity, after['v1'].disparity, equal_nan=True)
    assert np.array_equal(before['v2'].removed_mask, after['v2'].removed_mask)
    assert np.array_equal(before['v2'].rejected_disparity, after['v2'].rejected_disparity, equal_nan=True)
    assert 'interior_smoothing' in after['v2'].diagnostics


def test_cli_loads_nested_interior_config_and_old_positional_arguments(tmp_path):
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(interior=dict(enabled=True, windows=[31, 63]))))
    cfg = load_config(path)
    assert cfg.interior.enabled and cfg.interior.windows == (31, 63)
    assert RefinementConfig(cfg.alignment, 4.).threshold_px == 4.


@pytest.mark.parametrize('kwargs', [dict(enabled='yes'), dict(windows=(32,)), dict(windows=(31,31)),
    dict(strength=1.1), dict(minimum_coverage=0), dict(maximum_samples=64),
    dict(stereo_fit_p90_px=0), dict(mono_fit_p90=float('nan')), dict(fit_iterations=1), dict(guide_hole_radius=-1)])
def test_invalid_interior_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        InteriorSmoothConfig(**kwargs)
