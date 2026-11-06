"""One result layout and reader for all V1/V2 algorithms and mono sources."""
import json
from pathlib import Path
import re

import numpy as np

from .io import read_stereo_bin, write_q6
from .metrics import summarize_result
from .visualization import render_grid, write_gallery


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class ResultStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata = json.loads((self.root/'run.json').read_text())
        if self.metadata.get('schema_version') != 1:
            raise ValueError('unsupported result manifest schema')

    @property
    def frames(self):
        return self.metadata['frames']

    def _frame_path(self, stem, filename):
        if stem not in self.frames or not re.fullmatch(r'[A-Za-z0-9_-]+', stem):
            raise ValueError('unknown or invalid frame stem')
        path = (self.root/stem/filename).resolve()
        if self.root not in path.parents:
            raise ValueError('result path escapes run directory')
        return path

    def load(self, stem, variant):
        spec = self.metadata['variants'][variant]
        return np.load(self._frame_path(stem, spec['disparity']), allow_pickle=False)

    def load_original(self, stem):
        if stem not in self.frames or not re.fullmatch(r'[A-Za-z0-9_-]+', stem):
            raise ValueError('unknown or invalid frame stem')
        spec = self.metadata['original']
        if 'disparity' in spec:
            return np.load(self._frame_path(stem, spec['disparity']), allow_pickle=False)
        path = Path(spec['stereo_root'])/spec['file_pattern'].format(stem=stem)
        # Honor explicit historical import metadata without imposing it on new runs.
        return read_stereo_bin(path, *spec['shape'], spec['scale'],
                               undo_hflip=spec.get('undo_hflip', False))


def create_store(root, metadata, *, overwrite=False):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        if not overwrite or not (root/'run.json').is_file():
            raise ValueError('output is not empty; only an existing run may be explicitly overwritten')
        existing = json.loads((root/'run.json').read_text())
        if existing.get('schema_version') != 1:
            raise ValueError('refusing to overwrite an unrelated directory')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'run.json', dict(metadata, schema_version=1))
    return ResultStore(root)


def export_frame(store, stem, original, results, *, provenance=None):
    target = store.root/stem
    target.mkdir(exist_ok=True)
    np.save(target/'original_disparity_px.npy', original)
    reports = {}
    fields = {'Original':original}
    for variant, result in results.items():
        spec = store.metadata['variants'][variant]
        np.save(target/spec['disparity'], result.disparity)
        write_q6(target/f'{variant}_disp_q6.bin', result.disparity)
        np.save(target/f'{variant}_removed_mask.npy', result.removed_mask)
        np.save(target/f'{variant}_smoothed_mask.npy', result.smoothed_mask)
        if result.version == 'v2':
            np.save(target/f'{variant}_vote_only_disparity_px.npy', result.rejected_disparity)
            fields[f'{variant} vote only'] = result.rejected_disparity
        reports[variant] = dict(metrics=summarize_result(original, result), diagnostics=result.diagnostics)
        fields[variant] = result.disparity
    write_json(target/'summary.json', dict(stem=stem, variants=reports, provenance=provenance))
    render_grid(target/'full_frame_colormap_comparison.jpg', fields)
    return dict(stem=stem, variants=reports, provenance=provenance)


def finish_store(store, rows):
    totals = {}
    for row in rows:
        for variant, record in row['variants'].items():
            dest = totals.setdefault(variant, {})
            for name, value in record['metrics'].items():
                if isinstance(value, (float, int)):
                    dest[name] = (max(dest.get(name, 0), value) if name == 'maximum_change_px'
                                  else dest.get(name, 0)+value)
    write_json(store.root/'aggregate.json', dict(frames=[r['stem'] for r in rows], variants=totals,
        new_neural_inferences=0, warning='No independent ground truth; roughness is not accuracy.'))
    write_json(store.root/'per_frame.json', rows)
    write_gallery(store.root, [r['stem'] for r in rows], 'Stereo–Mono V1 / V2',
                  'Shared 10–192px scale. Black = invalid. No original holes are filled. See run.json for configuration.')


def write_comparison_report(root, summary):
    """One report format for arbitrary named runs and optional audit references."""
    lines = ['# Paired refinement comparison', '',
        f"Frames: {len(summary['frames'])}. Originally valid pixels: {summary['original_valid']:,}.", '',
        'All methods use identical original stereo. Roughness is evaluated only on common retained five-point stencils. '
        'Removal rates use all originally valid pixels; lower removal alone is not a quality improvement.', '',
        '| Method | Removed | Removal rate | Common-stencil roughness / px |',
        '|---|---:|---:|---:|']
    for name, value in summary['methods'].items():
        rough = value['common_mean_abs_laplacian_px']
        rough = f'{rough:.6f}' if rough is not None else 'N/A'
        lines += [f"| {name} | {value['removed']:,} | {100*value['removed_fraction']:.4f}% | {rough} |"]
    if summary.get('audit'):
        lines += ['', '## Optional cached-reference audit', '',
            'This reference is cached image matching, **not independent physical ground truth**. '
            'Supported means original-to-reference error ≤0.75px; conflicting means error >2px. '
            'Each removal rate uses its own original category as denominator. MAE uses identical locations retained by all methods.', '',
            '| Method | Supported removed | Supported removal rate | Conflicting removed | Conflict removal rate | Common-point MAE / px |',
            '|---|---:|---:|---:|---:|---:|']
        for name, value in summary['audit'].items():
            mae = value['paired_abs_error_sum']/value['paired_points'] if value['paired_points'] else None
            mae = f'{mae:.6f}' if mae is not None else 'N/A'
            lines += [f"| {name} | {value['removed_supported']:,} | "
                f"{100*value['removed_supported']/max(1,value['original_supported']):.4f}% | "
                f"{value['removed_conflicting']:,} | {100*value['removed_conflicting']/max(1,value['original_conflicting']):.4f}% | {mae} |"]
    if summary.get('pairs'):
        lines += ['', '## Pairwise changes', '',
            'Retained again means an originally valid sample rejected by the baseline was kept by the candidate. '
            'It is not filling an original hole or establishing that the sample is correct.', '',
            '| Baseline → candidate | Retained again | Newly removed | Maximum change on jointly retained points / px |',
            '|---|---:|---:|---:|']
        for value in summary['pairs']:
            lines += [f"| {value['baseline']} → {value['candidate']} | {value['retained_again']:,} | "
                f"{value['newly_removed']:,} | {value['maximum_common_change_px']:.6f} |"]
        for value in summary['pairs']:
            if 'retained_again_supported' in value:
                lines += ['', f"Audit of retained-again points for {value['candidate']}: "
                    f"{value['retained_again_supported']:,} supported, {value['retained_again_conflicting']:,} conflicting, "
                    f"{value['retained_again_intermediate']:,} intermediate, {value['retained_again_unchecked']:,} unchecked.", '']
    lines += ['', '## Galleries', '', '[All frames](index.html)', '']
    for stem in summary['frames']:
        lines += [f'- [{stem}: full frame]({stem}/full_frame_colormap_comparison.jpg)'
                  +(f' / [crop]({stem}/edge_comparison.jpg)' if (Path(root)/stem/'edge_comparison.jpg').exists() else '')]
    lines += ['', 'Exact inputs, variants, optional reference paths/hashes, and pairwise counts are recorded in [aggregate.json](aggregate.json).', '']
    (Path(root)/'REPORT.md').write_text('\n'.join(lines))
