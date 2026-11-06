from dataclasses import replace
import numpy as np
import pytest

from disp_fusion import CameraIntrinsics, DepthRgbCalibration, DepthRefiner, WindowConfig, RefinementConfig
from disp_fusion.calibration import project_depth_disparity_to_rgb
from disp_fusion.artifacts import create_store, export_frame, finish_store, ResultStore
from disp_fusion.io import write_q6


def calibration(height=90, width=110):
    intrinsics = CameraIntrinsics(100., 100., width/2, height/2, width, height)
    return DepthRgbCalibration(intrinsics, intrinsics, np.zeros(8), np.eye(3), np.zeros(3), 50., 0)


def example():
    y, x = np.indices((90, 110))
    mono = (x >= 55).astype(np.float32)+x*.0005
    stereo = (20+40*mono).astype(np.float32)
    stereo[30:60:4, 53] += 12
    stereo[10:15, 20:25] = np.nan
    return stereo, mono.astype(np.float32)


def test_many_versions_share_alignment_and_preserve_inputs(monkeypatch):
    import disp_fusion.pipeline as module
    original, mono = example(); source = original.copy(); prior = mono.copy()
    real = module.align_windows; calls = []
    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(module, 'align_windows', wrapped)
    refiner = DepthRefiner(calibration(), RefinementConfig(alignment=WindowConfig(window=65, stride=24)))
    result = refiner.refine_many(original, mono)
    assert len(calls) == 1
    assert np.array_equal(original, source, equal_nan=True) and np.array_equal(mono, prior)
    assert not (result['v2'].removed_mask & ~result['v1'].removed_mask).any()
    for name, item in result.items():
        kept = np.isfinite(item.disparity)
        assert not (kept & ~np.isfinite(source)).any()
        if name == 'v1':
            assert np.array_equal(item.disparity[kept], source[kept])
        else:
            assert np.max(np.abs(item.disparity[kept]-source[kept])) <= .75001


def test_empty_projection_and_all_invalid_input_are_noop():
    cal = calibration()
    original = np.full((90, 110), np.nan, np.float32)
    projection = project_depth_disparity_to_rgb(original, cal)
    assert (projection.source_flat_index == -1).all()
    for result in DepthRefiner(cal).refine_many(original, np.ones_like(original)).values():
        assert np.isnan(result.disparity).all() and not result.removed_mask.any()


def test_versions_shapes_and_q6_misuse_fail_clearly():
    refiner = DepthRefiner(calibration()); stereo, mono = example()
    with pytest.raises(ValueError, match='versions'):
        refiner.refine(stereo, mono, 'v7')
    with pytest.raises(ValueError, match='floating-point'):
        refiner.refine(np.zeros_like(stereo, np.int16), mono)
    with pytest.raises(ValueError, match='stereo shape'):
        refiner.refine(stereo[:-1], mono)
    bad = mono.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        refiner.refine(stereo, bad)


@pytest.mark.parametrize('kwargs', [dict(threshold_px=-1), dict(vote_fraction=.5),
    dict(max_change_px=-1), dict(smooth_iterations=1.5), dict(mono_sigma=0)])
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RefinementConfig(**kwargs)


def test_result_store_roundtrip_and_no_unrelated_overwrite(tmp_path):
    original, mono = example()
    result = DepthRefiner(calibration()).refine(original, mono, 'v1')
    metadata = dict(frames=['000000'], variants={'v1':dict(disparity='v1.npy')},
                    original=dict(disparity='original_disparity_px.npy'))
    store = create_store(tmp_path/'run', metadata)
    row = export_frame(store, '000000', original, {'v1':result})
    finish_store(store, [row])
    loaded = ResultStore(tmp_path/'run')
    assert np.array_equal(loaded.load('000000', 'v1'), result.disparity, equal_nan=True)
    assert np.array_equal(loaded.load_original('000000'), original, equal_nan=True)
    assert (store.root/'index.html').is_file()
    unrelated = tmp_path/'unrelated'; unrelated.mkdir(); (unrelated/'keep.txt').write_text('keep')
    with pytest.raises(ValueError):
        create_store(unrelated, metadata, overwrite=True)
    assert (unrelated/'keep.txt').read_text() == 'keep'


def test_q6_export_rejects_overflow_instead_of_clipping(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        write_q6(tmp_path/'bad.bin', np.array([[600.]], np.float32))


@pytest.mark.parametrize('kwargs', [dict(smooth_strength=0), dict(smooth_iterations=0), dict(max_change_px=0)])
def test_disabled_smoothing_is_bit_exact_for_float64(kwargs):
    from disp_fusion.smoothing import bounded_guided_smooth
    source = np.arange(120, dtype=np.float64).reshape(10, 12)/37+20
    source[0, 0] = np.nan
    result, report = bounded_guided_smooth(source, np.zeros_like(source), np.ones_like(source, bool),
                                          RefinementConfig(**kwargs))
    assert np.array_equal(result, source, equal_nan=True)
    assert report['maximum_change_px'] == 0
