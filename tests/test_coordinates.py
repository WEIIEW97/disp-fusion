"""Normal camera coordinates are preserved through import and RGB mapping."""
from dataclasses import asdict, replace
import hashlib
import json

import cv2
import numpy as np
import pytest

from disp_fusion import DepthRefiner
from disp_fusion.alignment import reject_from_rgb
from disp_fusion.artifacts import create_store
from disp_fusion.calibration import project_depth_disparity_to_rgb
from disp_fusion.cli import main
from disp_fusion.io import read_stereo_bin, write_q6
from test_api import calibration, example


def test_disparity_reader_preserves_asymmetric_stored_orientation(tmp_path):
    raw = np.array([[64, 128, 192], [256, -1, 384]], np.int16)
    path = tmp_path/'disparity.bin'
    raw.tofile(path)
    expected = np.array([[1., 2., 3.], [4., np.nan, 6.]], np.float32)
    actual = read_stereo_bin(path, 2, 3)
    assert np.array_equal(actual, expected, equal_nan=True)


def test_legacy_reader_conversion_requires_explicit_request(tmp_path):
    raw = np.array([[64, 128, 192], [256, 320, 384]], np.int16)
    path = tmp_path/'disparity.bin'
    raw.tofile(path)
    normal = read_stereo_bin(path, 2, 3)
    legacy = read_stereo_bin(path, 2, 3, undo_hflip=True)
    assert np.array_equal(legacy, normal[:, ::-1])


@pytest.mark.parametrize('legacy_orientation', [None, False, True])
def test_result_store_uses_only_explicit_historical_orientation(tmp_path, legacy_orientation):
    raw = np.array([[64, 128, 192], [256, 320, 384]], np.int16)
    raw.tofile(tmp_path/'000000_disp.bin')
    spec = dict(stereo_root=str(tmp_path), file_pattern='{stem}_disp.bin',
                shape=[2, 3], scale=64)
    if legacy_orientation is not None:
        spec['undo_hflip'] = legacy_orientation
    store = create_store(tmp_path/'results', dict(frames=['000000'], variants={}, original=spec))
    expected = raw.astype(np.float32)/64
    if legacy_orientation is True:
        expected = expected[:, ::-1]
    assert np.array_equal(store.load_original('000000'), expected)


def test_identity_rgb_projection_keeps_asymmetric_source_indices():
    cal = calibration(height=3, width=6)
    stereo = np.arange(18, dtype=np.float32).reshape(3, 6)+20
    stereo[0, 1] = np.nan
    projection = project_depth_disparity_to_rgb(stereo, cal)
    expected = np.arange(18).reshape(3, 6)
    expected[0, 1] = -1
    assert np.array_equal(projection.source_flat_index, expected)
    np.testing.assert_allclose(projection.equivalent_disparity, stereo, rtol=1e-6)


def test_rgb_rejection_returns_to_calibrated_source_not_mirrored_pixel():
    cal = replace(calibration(height=3, width=6),
                  depth_to_rgb_translation_mm=np.array([2.5, 0., 0.]))
    stereo = np.full((3, 6), 20., np.float32)
    stereo[1, 1] = np.nan
    projection = project_depth_disparity_to_rgb(stereo, cal)
    # Z=250mm, fx=100px, tx=2.5mm: RGB receives a one-pixel right shift.
    assert projection.source_flat_index[1, 4] == 1*6+3
    assert (projection.source_flat_index[:, 0] == -1).all()
    residual = np.zeros_like(stereo)
    residual[1, 4] = 10
    output, removed = reject_from_rgb(stereo, projection.source_flat_index,
                                     residual, np.ones_like(stereo, bool), 4)
    expected = np.zeros_like(stereo, bool)
    expected[1, 3] = True
    assert np.array_equal(removed, expected)
    assert np.isnan(output[1, 3]) and np.isnan(output[1, 1])
    assert np.array_equal(output[:, -1], stereo[:, -1])
    kept = np.isfinite(output)
    assert np.array_equal(output[kept], stereo[kept])


def test_cli_normal_inputs_match_array_api_without_orientation_conversion(tmp_path):
    original, mono = example()
    original = np.round(original*64)/64
    height, width = original.shape
    cal = calibration(height, width)
    data = tmp_path/'data'
    rgb_dir = data/'RGB'
    rgb_dir.mkdir(parents=True)
    rgb_path = rgb_dir/'000000.png'
    assert cv2.imwrite(str(rgb_path), np.zeros((height, width, 3), np.uint8))
    cache = tmp_path/'mono'
    cache.mkdir()
    np.save(cache/'000000.npy', mono)
    (cache/'run_info.json').write_text(json.dumps(dict(model='synthetic', input_subdir='RGB',
        undo_input_hflip=False, inputs=[dict(stem='000000',
        input_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest())])))
    stereo_dir = tmp_path/'stereo'
    stereo_dir.mkdir()
    write_q6(stereo_dir/'000000_disp.bin', original)
    (data/'raw_calibration_list.json').write_text(json.dumps(dict(baseline_mm=cal.baseline_mm,
        params=[dict(index=0, depth_intrinsic=asdict(cal.depth), rgb_intrinsic=asdict(cal.rgb),
            rgb_distortion=dict.fromkeys(['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'], 0.),
            transform=dict(rot=cal.depth_to_rgb_rotation.tolist(),
                           trans=cal.depth_to_rgb_translation_mm.tolist()))])))
    output = tmp_path/'output'
    main(['run', '--data-root', str(data), '--stereo-root', str(stereo_dir),
          '--mono-cache', str(cache), '--output', str(output), '--all',
          '--width', str(width), '--height', str(height)])
    saved = np.load(output/'000000'/'original_disparity_px.npy')
    assert np.array_equal(saved, original, equal_nan=True)
    expected = DepthRefiner(cal).refine_many(original, mono)
    for version, result in expected.items():
        actual = np.load(output/'000000'/f'{version}_tau4_disparity_px.npy')
        assert np.array_equal(actual, result.disparity, equal_nan=True)
    metadata = json.loads((output/'run.json').read_text())
    assert metadata['coordinate_system'] == 'input stereo-reference; native RGB mono; disparity px'
    replay_output = tmp_path/'replay_output'
    main(['run', '--data-root', str(data), '--stereo-run', str(output),
          '--mono-cache', str(cache), '--output', str(replay_output), '--all',
          '--width', str(width), '--height', str(height)])
    for version, result in expected.items():
        actual = np.load(replay_output/'000000'/f'{version}_tau4_disparity_px.npy')
        assert np.array_equal(actual, result.disparity, equal_nan=True)
