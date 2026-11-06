import numpy as np

from disp_fusion.config import RefinementConfig
from disp_fusion.smoothing import bounded_guided_smooth
from disp_fusion.pipeline import consensus_mask, source_mask


def test_consensus_requires_count_and_same_direction_and_v1_flag():
    evidence = dict(vote_count=np.array([[5, 5, 2, 5]]),
        positive_votes=np.array([[4, 2, 2, 5]]), negative_votes=np.array([[0, 3, 0, 0]]),
        edge_band=np.ones((1, 4), bool), residual_px=np.array([[6., 6., 6., 1.]]))
    mask, _ = consensus_mask(evidence)
    assert mask.tolist() == [[True, False, False, False]]


def test_source_mapping_never_activates_original_holes():
    original = np.array([[20., np.nan], [0., 40.]], np.float32)
    index = np.array([[0, 1], [2, -1]])
    mask = source_mask(original, index, np.ones_like(index, bool))
    assert mask.tolist() == [[True, False], [False, False]]


def test_guided_smoothing_reduces_flat_noise_without_filling_holes():
    rng = np.random.default_rng(843)
    stereo = (30+rng.normal(0, .2, (90, 110))).astype(np.float32)
    stereo[20:30, 40:60] = np.nan
    guide = np.ones_like(stereo)
    output, report = bounded_guided_smooth(stereo, guide, np.ones_like(stereo, bool))
    valid = np.isfinite(stereo)
    assert np.array_equal(np.isfinite(output), valid)
    assert np.std(output[valid]) < .5*np.std(stereo[valid])
    assert np.abs(output[valid]-stereo[valid]).max() <= .75001
    assert report['changed_over_005px'] > 0


def test_mono_edge_and_stereo_jump_block_cross_surface_smoothing():
    stereo = np.full((50, 80), 20., np.float32)
    stereo[:, 40:] = 60
    guide = np.zeros_like(stereo); guide[:, 40:] = 1
    output, _ = bounded_guided_smooth(stereo, guide, np.ones_like(stereo, bool))
    assert np.array_equal(output, stereo)


def test_mono_alone_blocks_small_stereo_step():
    stereo = np.full((30, 50), 30., np.float32)
    stereo[:, 25:] = 31.
    guide = np.zeros_like(stereo); guide[:, 25:] = 1.
    output, _ = bounded_guided_smooth(stereo, guide, np.ones_like(stereo, bool))
    assert np.array_equal(output, stereo)


def test_adding_vote_counts_does_not_change_legacy_alignment():
    from disp_fusion.alignment import align_windows
    from disp_fusion.config import WindowConfig
    y, x = np.indices((90, 110))
    mono = (x >= 55).astype(np.float32)+x*.0005
    stereo = (20+40*mono).astype(np.float32)
    cfg = WindowConfig(window=65, stride=24)
    old = align_windows(stereo, mono, cfg)
    new = align_windows(stereo, mono, cfg, vote_threshold_px=4)
    for key in ['aligned_mono', 'residual_px', 'edge_band', 'anchors']:
        assert np.array_equal(old[key], new[key], equal_nan=True)
    assert new['vote_count'].max() >= 3


def test_unsafe_pixels_are_immutable():
    stereo = np.arange(120, dtype=np.float32).reshape(10, 12)*.01+30
    safe = np.ones_like(stereo, bool); safe[3:6, 4:8] = False
    output, _ = bounded_guided_smooth(stereo, np.zeros_like(stereo), safe)
    assert np.array_equal(output[~safe], stereo[~safe])


def test_cap_holds_and_zero_strength_is_identity():
    rng = np.random.default_rng(78)
    stereo = (30+rng.normal(0, .5, (40, 50))).astype(np.float32)
    safe = np.ones_like(stereo, bool)
    out, _ = bounded_guided_smooth(stereo, np.zeros_like(stereo), safe,
                                 RefinementConfig(max_change_px=.1))
    assert np.max(np.abs(out-stereo)) < .10001
    out, _ = bounded_guided_smooth(stereo, np.zeros_like(stereo), safe,
                                 RefinementConfig(smooth_strength=0))
    assert np.array_equal(out, stereo)
