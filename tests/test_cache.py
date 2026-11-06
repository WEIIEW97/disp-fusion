import pytest

from disp_fusion.cache import check_cached_rgb


def test_legacy_cache_requires_explicit_opt_in_and_records_limitation():
    meta = dict(input_subdir='RGB', undo_input_hflip=False)
    with pytest.raises(ValueError, match='historical input hash absent'):
        check_cached_rgb(meta, '000286', 'abc')
    result = check_cached_rgb(meta, '000286', 'abc', True)
    assert not result['historical_input_identity_verified']
    assert result['limitation'] is not None


def test_mismatched_hash_is_not_bypassed_by_legacy_flag():
    meta = dict(input_subdir='RGB', undo_input_hflip=False,
                inputs=[dict(stem='000286', input_sha256='abc')])
    with pytest.raises(ValueError, match='differs'):
        check_cached_rgb(meta, '000286', 'def', True)
    assert check_cached_rgb(meta, '000286', 'abc')['historical_input_identity_verified']


def test_flipped_or_ir_cache_is_rejected():
    for meta in [dict(input_subdir='RightIR', undo_input_hflip=False),
                 dict(input_subdir='RGB', undo_input_hflip=True)]:
        with pytest.raises(ValueError, match='native stored-RGB'):
            check_cached_rgb(meta, '000286', 'abc', True)
