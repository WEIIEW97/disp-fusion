import json
import numpy as np

from disp_fusion import DepthRefiner
from disp_fusion.artifacts import create_store, export_frame, finish_store
from disp_fusion.cli import main
from test_api import calibration, example


def test_unified_compare_reads_self_contained_runs(tmp_path):
    original, mono = example()
    result = DepthRefiner(calibration()).refine_many(original, mono)
    for name in ['a', 'b']:
        metadata = dict(frames=['000000'], variants={'v1':dict(disparity='v1.npy'),
                        'v2':dict(disparity='v2.npy')}, original=dict(disparity='original_disparity_px.npy'))
        store = create_store(tmp_path/name, metadata)
        row = export_frame(store, '000000', original, result)
        finish_store(store, [row])
    audit = tmp_path/'audit'/'000000'; audit.mkdir(parents=True)
    np.savez(audit/'dense_evidence.npz', disparity=original, center_checked=np.isfinite(original))
    main(['compare', '--input', 'first', str(tmp_path/'a'), 'v1',
          '--input', 'second', str(tmp_path/'b'), 'v2', '--output', str(tmp_path/'comparison'),
          '--audit-cache', str(tmp_path/'audit'), '--pair', 'first', 'second',
          '--crop', '40', '20', '80', '60'])
    summary = json.loads((tmp_path/'comparison'/'aggregate.json').read_text())
    assert summary['common_retained'] > 0
    assert set(summary['methods']) == {'Original', 'first', 'second'}
    assert (tmp_path/'comparison'/'000000'/'full_frame_colormap_comparison.jpg').is_file()
    assert (tmp_path/'comparison'/'000000'/'edge_comparison.jpg').is_file()
    assert (tmp_path/'comparison'/'000000'/'pair_0_change_map.png').is_file()
    assert (tmp_path/'comparison'/'REPORT.md').is_file()
    pair = summary['pairs'][0]
    assert pair['newly_removed'] == 0
    assert pair['retained_again'] == pair['retained_again_supported']
    assert summary['audit']['Original']['removed_supported'] == 0
