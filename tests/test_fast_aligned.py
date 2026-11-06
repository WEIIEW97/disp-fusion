"""Portable native backend: bounds, masks, edges, planes and reusable buffers."""
import ctypes as ct
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from disp_fusion import FastAlignedConfig, FastAlignedRefiner
from disp_fusion.fast_aligned import _Config


@pytest.fixture(scope='session', params=['g++', 'clang++', 'scalar'])
def native_library(tmp_path_factory, request):
    compiler = 'g++' if request.param == 'scalar' else request.param
    if not shutil.which('cmake') or not shutil.which(compiler):
        pytest.skip('native tests require CMake and a C++17 compiler')
    target = tmp_path_factory.mktemp('fast-aligned-build')
    source = Path(__file__).resolve().parents[1]/'native'
    options = [f'-DCMAKE_CXX_COMPILER={compiler}']
    if request.param == 'scalar':
        options += ['-DDF_FAST_SIMD=OFF', '-DDF_FAST_VECTOR_TYPES=OFF']
    elif request.param == 'clang++':
        options += ['-DDF_FAST_NATIVE_TUNING=ON']
    subprocess.run(['cmake', '-S', str(source), '-B', str(target), '-DCMAKE_BUILD_TYPE=Release', *options], check=True, capture_output=True)
    subprocess.run(['cmake', '--build', str(target), '-j', '2'], check=True, capture_output=True)
    return target/'libdisp_fusion_fast.so'


def wall(height=128, width=160):
    y, x = np.indices((height, width))
    truth = 30+.01*x+.005*y
    rng = np.random.default_rng(19)
    stereo = (truth+.18*np.sin(2*np.pi*x/12)+rng.normal(0, .025, truth.shape)).astype(np.float32)
    mono = (.2+.0002*x+.0001*y).astype(np.float32)
    return truth, stereo, mono


@pytest.mark.parametrize('tile',[8,16,32])
def test_native_reduces_stripes_and_keeps_the_plane(native_library,tile):
    truth, stereo, mono = wall()
    with FastAlignedRefiner(160, 128, FastAlignedConfig(tile_size=tile), library=native_library) as refiner:
        result = refiner.refine(stereo, mono)
    roi = np.s_[24:-24, 24:-24]
    assert np.std((result.disparity-truth)[roi]) < .5*np.std((stereo-truth)[roi])
    y, x = np.indices(stereo.shape)
    beta = np.linalg.lstsq(np.c_[x[roi].ravel(), y[roi].ravel(), np.ones(x[roi].size)], result.disparity[roi].ravel(), rcond=None)[0]
    np.testing.assert_allclose(beta[:2], [.01, .005], atol=.0005)
    assert abs(np.median((result.disparity-stereo)[roi])) < .025
    assert result.diagnostics['native']['plane_models'] > 0
    assert np.max(abs(result.disparity-stereo)) <= .75002


@pytest.mark.parametrize('edge_mode',[1,2])
def test_invalid_and_explicitly_protected_samples_are_bit_exact(native_library,edge_mode):
    _, stereo, mono = wall()
    stereo[10, :5] = [np.nan, 0, -1, np.inf, -np.inf]
    mono[60:62, 70:74] = np.nan
    protected = np.zeros(stereo.shape, bool); protected[40:45, 30:35] = True
    frozen = protected | ~np.isfinite(mono) | ~np.isfinite(stereo) | (stereo <= .05)
    with FastAlignedRefiner(160, 128, FastAlignedConfig(edge_mode=edge_mode), library=native_library) as refiner:
        result = refiner.refine(stereo, mono, protected_mask=protected)
    assert np.array_equal(result.disparity.view(np.uint32)[frozen], stereo.view(np.uint32)[frozen])
    assert not result.removed_mask[frozen].any()
    valid = np.isfinite(result.disparity)&(result.disparity > .05)
    assert not (valid & (~np.isfinite(stereo) | (stereo <= .05))).any()


def test_step_stays_sharp_and_sparse_edge_spikes_can_be_invalidated(native_library):
    y, x = np.indices((128,160))
    mono = ((x >= 80)+.0005*x).astype(np.float32)
    original = (20+40*mono).astype(np.float32)
    stereo = original.copy(); stereo[28:104:4,78] += 12
    with FastAlignedRefiner(160,128,library=native_library) as refiner:
        result = refiner.refine(stereo,mono)
    assert result.removed_mask[28:104:4,78].sum() >= 10
    correct = np.ones(stereo.shape,bool); correct[28:104:4,78] = False
    correct &= np.abs(x-80) <= 2
    assert np.array_equal(result.disparity[correct], original[correct])


def test_coherent_curved_mono_does_not_authorize_large_plane_smoothing(native_library):
    _, stereo, _ = wall()
    y, x = np.indices(stereo.shape)
    mono = (((x-80)/20)**2+((y-64)/20)**2).astype(np.float32)
    cfg = FastAlignedConfig(mono_plane_rms=.00001)
    with FastAlignedRefiner(160,128,cfg,library=native_library) as refiner:
        result = refiner.refine(stereo,mono)
    assert result.diagnostics['native']['plane_models'] == 0
    assert np.array_equal(result.disparity,stereo)


def test_preallocated_process_is_repeatable_and_rejects_aliases(native_library):
    _, stereo, mono = wall()
    output = np.empty_like(stereo); removed = np.empty(stereo.shape,np.uint8); smooth = np.empty_like(removed)
    with FastAlignedRefiner(160,128,library=native_library) as refiner:
        workspace = refiner.workspace_bytes
        a = refiner.process_into(stereo,mono,output,removed,smooth)
        expected = output.copy()
        b = refiner.process_into(stereo,mono,output,removed,smooth)
        assert np.array_equal(expected,output) and a == b
        assert refiner.workspace_bytes == workspace
        with pytest.raises(ValueError,match='overlap'):
            refiner.process_into(stereo,mono,stereo,removed,smooth)
    with pytest.raises(RuntimeError,match='closed'):
        refiner.refine(stereo,mono)


@pytest.mark.parametrize('shape',[(2,2),(7,5),(33,67),(400,640)])
def test_borders_partial_tiles_and_original_holes(native_library,shape):
    h,w = shape
    stereo = np.full(shape,30,np.float32);mono = np.ones(shape,np.float32)
    stereo[h//2,w//2] = np.nan
    with FastAlignedRefiner(w,h,library=native_library) as refiner:
        result = refiner.refine(stereo,mono)
    assert np.array_equal(result.disparity,stereo,equal_nan=True)


def test_all_unusable_mono_and_zero_change_are_safe(native_library):
    _,stereo,mono = wall()
    with FastAlignedRefiner(160,128,replace(FastAlignedConfig(),max_change_px=0),library=native_library) as refiner:
        assert np.array_equal(refiner.refine(stereo,mono).disparity,stereo)
        result = refiner.refine(stereo,np.full_like(mono,np.nan))
        assert np.array_equal(result.disparity,stereo)
        assert result.diagnostics['native']['plane_models'] == 0


def test_masked_plane_with_large_offset_remains_numerically_stable(native_library):
    rng = np.random.default_rng(1703)
    y,x = np.indices((128,160))
    truth = 512+.003*x+.002*y
    stereo = (truth+rng.normal(0,.08,truth.shape)).astype(np.float32)
    mono = (.1+.0002*x+.0001*y).astype(np.float32)
    stereo[rng.random(truth.shape)<.04] = np.nan
    mono[rng.random(truth.shape)<.08] = np.nan
    with FastAlignedRefiner(160,128,library=native_library) as refiner:
        result = refiner.refine(stereo,mono)
    core = np.zeros(truth.shape,bool);core[24:-24,24:-24] = True
    common = core&np.isfinite(stereo)&np.isfinite(mono)&np.isfinite(result.disparity)
    assert np.std((result.disparity-truth)[common]) < .65*np.std((stereo-truth)[common])
    assert abs(np.median((result.disparity-truth)[common])) < .015
    assert np.array_equal(result.disparity[~np.isfinite(mono)],stereo[~np.isfinite(mono)],equal_nan=True)


def test_native_and_python_configuration_defaults_agree(native_library):
    lib = ct.CDLL(str(native_library));lib.df_fast_default_config.argtypes = [ct.POINTER(_Config)]
    config = _Config();lib.df_fast_default_config(ct.byref(config))
    python = FastAlignedConfig()
    for name,_ in _Config._fields_:
        assert getattr(config,name) == pytest.approx(getattr(python,name),rel=1e-6)


@pytest.mark.parametrize('kwargs',[dict(tile_size=15),dict(minimum_votes=5),dict(max_change_px=-1),
    dict(mono_edge=float('nan')),dict(edge_band_px=1),dict(tile_size=True),dict(vote_fraction=.5),
    dict(edge_mode=3),dict(edge_samples=15),dict(edge_samples=257),dict(edge_min_samples=65),
    dict(edge_iterations=0),dict(edge_window=64),dict(edge_window=65),dict(edge_stride=7),
    dict(edge_fit_p80_px=float('nan')),dict(edge_anchor_guard_px=17)])
def test_bad_fast_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FastAlignedConfig(**kwargs)


def test_wrapper_does_not_hide_conversions_or_missing_builds(native_library,tmp_path):
    with pytest.raises(FileNotFoundError,match='Build'):
        FastAlignedRefiner(160,128,library=tmp_path/'missing.so')
    _,stereo,mono = wall()
    with FastAlignedRefiner(160,128,library=native_library) as refiner:
        with pytest.raises(ValueError,match='float32'):
            refiner.refine(stereo.astype(np.float64),mono)
        with pytest.raises(ValueError,match='contiguous'):
            refiner.refine(np.asfortranarray(stereo),mono)
        buffer = np.zeros(stereo.nbytes+1,np.uint8)
        unaligned = np.ndarray(stereo.shape,np.float32,buffer=buffer,offset=1)
        with pytest.raises(ValueError,match='aligned'):
            refiner.refine(unaligned,mono)


def test_edge_rejection_is_independent_of_secondary_plane_stage(native_library):
    y,x = np.indices((400,640))
    mono = ((x>=320)+.0004*x+.0001*y).astype(np.float32)
    stereo = (20+40*mono+.12*np.sin(x/2)).astype(np.float32)
    spikes = (x==317)&(y%4==0)&(y>40)&(y<360)
    stereo[spikes] += 12
    stereo[100:110,100:110] = np.nan
    masks = []
    for kwargs in [dict(smooth_strength=0), dict(tile_size=16), dict(tile_size=32), dict(max_change_px=0)]:
        with FastAlignedRefiner(640,400,FastAlignedConfig(**kwargs),library=native_library) as engine:
            result = engine.refine(stereo,mono)
        masks.append(result.removed_mask)
        retained_edge = (abs(x-320)<=6)&~result.removed_mask
        np.testing.assert_array_equal(result.disparity[retained_edge],stereo[retained_edge])
        assert not np.isfinite(result.disparity[100:110,100:110]).any()
    assert masks[0][spikes].sum() >= .9*spikes.sum()
    assert not masks[0][~spikes].any()
    for mask in masks[1:]:
        np.testing.assert_array_equal(mask,masks[0])


def test_clean_affine_step_is_not_deleted(native_library):
    y,x = np.indices((400,640))
    mono = ((x>=320)+.0005*x+.0001*y).astype(np.float32)
    stereo = (20+40*mono).astype(np.float32)
    with FastAlignedRefiner(640,400,library=native_library) as engine:
        result = engine.refine(stereo,mono)
    assert not result.removed_mask.any()
    np.testing.assert_array_equal(result.disparity[:,312:329],stereo[:,312:329])


def test_spatial_consensus_guard_preserves_offset_slanted_edge_and_removes_spikes(native_library):
    y,x=np.indices((360,640));boundary=250+.31*y
    mono=((x>=boundary-1)+.0003*x+.0001*y).astype(np.float32)
    truth=(20+35*(x>=boundary)+.006*x+.003*y).astype(np.float32)
    spikes=(x==np.ceil(boundary).astype(int)+3)&(y%7==0)&(y>30)&(y<330)
    stereo=truth.copy();stereo[spikes]+=14
    base=FastAlignedConfig(reject_threshold_px=2,edge_fit_p80_px=1,stereo_jump_px=1,anchor_deviation_px=.375)
    with FastAlignedRefiner(640,360,base,library=native_library) as engine:
        old=engine.refine(stereo,mono)
    with FastAlignedRefiner(640,360,replace(base,edge_mode=2),library=native_library) as engine:
        guarded=engine.refine(stereo,mono)
    assert (old.removed_mask&~spikes).sum()>=300
    assert not (guarded.removed_mask&~spikes).any()
    assert guarded.removed_mask[spikes].all()
    assert not (guarded.removed_mask&~old.removed_mask).any()
    # Interior plane results are unaffected, and rescued original-valid pixels
    # retain their original disparity exactly; no interpolation fills an edge.
    np.testing.assert_array_equal(guarded.disparity[~old.removed_mask],old.disparity[~old.removed_mask])
    rescued=old.removed_mask&~guarded.removed_mask
    np.testing.assert_array_equal(guarded.disparity[rescued],stereo[rescued])


def test_guarded_rejection_remains_independent_of_plane_settings(native_library):
    y,x=np.indices((360,640));mono=((x>=320)+.0004*x+.0001*y).astype(np.float32)
    stereo=(20+40*mono+.12*np.sin(x/2)).astype(np.float32)
    stereo[40:320:4,317]+=12;stereo[100:110,100:110]=np.nan;mono[230:240,318:320]=np.nan
    masks=[]
    for options in [dict(),dict(smooth_strength=0),dict(max_change_px=0),dict(tile_size=32)]:
        with FastAlignedRefiner(640,360,FastAlignedConfig(edge_mode=2,**options),library=native_library) as engine:
            result=engine.refine(stereo,mono)
        masks.append(result.removed_mask)
        assert not np.isfinite(result.disparity[100:110,100:110]).any()
        np.testing.assert_array_equal(result.disparity[~np.isfinite(mono)],stereo[~np.isfinite(mono)])
    assert masks[0].sum()>0
    for mask in masks[1:]:np.testing.assert_array_equal(mask,masks[0])


def test_guarded_spikes_near_image_border_are_safe(native_library):
    # The continuous component supplies mono span even though the anchor guard
    # excludes the very narrow background strip beside the image boundary.
    y,x=np.indices((360,640));mono=((x>=2)+.003*x+.001*y).astype(np.float32)
    stereo=(20+40*mono).astype(np.float32);stereo[40:320:7,0]+=14
    protected=np.zeros(stereo.shape,np.uint8);protected[100:120,:2]=1
    mono[170:180,1]=np.nan
    with FastAlignedRefiner(640,360,FastAlignedConfig(edge_mode=2),library=native_library) as engine:
        result=engine.refine(stereo,mono,protected_mask=protected)
    frozen=(protected!=0)|~np.isfinite(mono)
    np.testing.assert_array_equal(result.disparity[frozen],stereo[frozen])
    assert result.removed_mask[:,0].sum()>10
    assert not result.removed_mask[:,2:].any()


def test_guard_does_not_bridge_missing_depth_hypotheses(native_library):
    y,x=np.indices((360,640));boundary=250+.31*y
    mono=((x>=boundary)+.0003*x+.0001*y).astype(np.float32)
    truth=(20+35*(x>=boundary)+.006*x+.003*y).astype(np.float32)
    flying=x==np.ceil(boundary).astype(int)-1
    stereo=truth.copy();stereo[flying]+=17.5
    cfg=FastAlignedConfig(edge_mode=2,reject_threshold_px=2,edge_fit_p80_px=1,
        stereo_jump_px=1,anchor_deviation_px=.375)
    with FastAlignedRefiner(640,360,cfg,library=native_library) as engine:
        result=engine.refine(stereo,mono)
    # The two sampled surfaces bracket this depth, but no nearby guidance
    # hypothesis predicts it. A min/max interval would incorrectly retain it.
    assert result.removed_mask[flying].all()
    assert not result.removed_mask[~flying].any()
