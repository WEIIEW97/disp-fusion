"""Opt-in real-data regression: both versions and both mono models."""
import os
from pathlib import Path
import numpy as np
import pytest

from disp_fusion import DepthRefiner, InteriorSmoothConfig, RefinementConfig, load_depth_rgb_calibration
from disp_fusion.artifacts import ResultStore
from disp_fusion.cache import CachedMonoSource

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/william/extdisk/data/data2hisi/original/TGT_20260930_134305')
FRAMES = '000000 000050 000100 000150 000200 000250 000286 000300 000350 000400 000450 000500 000550 000600 000650 000700 000750 000800 000850 000900'.split()
pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('DISP_FUSION_REAL_DATA') != '1',
              reason='set DISP_FUSION_REAL_DATA=1 to replay saved real-data results')]


@pytest.mark.parametrize('model', ['dam518', 'zipdepth'])
@pytest.mark.parametrize('stem', FRAMES)
def test_saved_v1_v2_are_bit_identical(model, stem):
    v1 = ResultStore(ROOT/'outputs'/f'simple_edge_rejection_{model}_sample20')
    v2 = ResultStore(ROOT/'outputs'/f'edge_consensus_smooth_v2_{model}_sample20')
    cache = CachedMonoSource(v1.metadata['mono_cache'], allow_legacy_unhashed=model == 'zipdepth')
    mono, _ = cache.load(stem, DATA/'RGB'/f'{stem}.png')
    original = v1.load_original(stem)
    cal = load_depth_rgb_calibration(DATA/'raw_calibration_list.json', (1280, 720), (1280, 720))
    results = DepthRefiner(cal).refine_many(original, mono)
    assert np.array_equal(results['v1'].disparity, v1.load(stem, 'v1_tau4'), equal_nan=True)
    assert np.array_equal(results['v2'].disparity, v2.load(stem, 'v2_tau4'), equal_nan=True)
    assert np.array_equal(results['v2'].rejected_disparity, v2.load(stem, 'v2_vote_tau4'), equal_nan=True)


@pytest.mark.parametrize('model', ['dam518', 'zipdepth'])
@pytest.mark.parametrize('stem', ['000286', '000450', '000900'])
def test_saved_enhanced_output_and_masks_are_bit_identical(model, stem):
    baseline = ResultStore(ROOT/'outputs'/f'edge_consensus_smooth_v2_{model}_sample20')
    enhanced = ResultStore(ROOT/'outputs/interior_multiscale_sample20_20261029'/model)
    cache = CachedMonoSource(baseline.metadata['mono_cache'], allow_legacy_unhashed=model == 'zipdepth')
    mono, _ = cache.load(stem, DATA/'RGB'/f'{stem}.png')
    original = baseline.load_original(stem)
    cal = load_depth_rgb_calibration(DATA/'raw_calibration_list.json', (1280, 720), (1280, 720))
    cfg = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
    result = DepthRefiner(cal, cfg).refine(original, mono)
    assert np.array_equal(result.disparity, enhanced.load(stem, 'v2_tau4'), equal_nan=True)
    for name in ['removed_mask', 'smoothed_mask']:
        expected = np.load(enhanced.root/stem/f'v2_tau4_{name}.npy')
        assert np.array_equal(getattr(result, name), expected)
