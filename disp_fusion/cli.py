"""One CLI for cached-mono V1/V2 processing and arbitrary run comparison."""
import argparse
from dataclasses import asdict, replace
import json
import hashlib
from pathlib import Path

import numpy as np
import cv2

from .alignment import valid_disparity
from .artifacts import ResultStore, create_store, export_frame, finish_store, write_json, write_comparison_report
from .cache import CachedMonoSource
from .calibration import load_depth_rgb_calibration
from .config import WindowConfig, InteriorSmoothConfig, RefinementConfig
from .io import read_stereo_bin
from .metrics import five_point_mask, laplacian, audit_against_reference
from .pipeline import DepthRefiner
from .visualization import render_comparison_grid, write_gallery, colorize_disparity


SAMPLE20 = '000000 000050 000100 000150 000200 000250 000286 000300 000350 000400 000450 000500 000550 000600 000650 000700 000750 000800 000850 000900'.split()


def load_config(path):
    if path is None:
        return RefinementConfig()
    values = json.loads(Path(path).read_text())
    return RefinementConfig(**dict(values, alignment=WindowConfig(**values.get('alignment', {})),
                                  interior=InteriorSmoothConfig(**values.get('interior', {}))))


def run_command(args):
    cache = CachedMonoSource(args.mono_cache, allow_legacy_unhashed=args.allow_legacy_unhashed_cache)
    frames = args.frames or (SAMPLE20 if args.sample20 else cache.frames)
    if not frames or len(set(frames)) != len(frames):
        raise ValueError('frames must be nonempty and unique')
    if not set(frames) <= set(cache.frames):
        raise ValueError('requested frames are not all present in the cache')
    output = args.output.resolve()
    stereo_source = args.stereo_root if args.stereo_root is not None else args.stereo_run
    for source in [cache.root, args.data_root.resolve(), stereo_source.resolve()]:
        if output == source or source in output.parents:
            raise ValueError('output must not modify an input directory')
    config = load_config(args.config)
    source_store = ResultStore(args.stereo_run) if args.stereo_run is not None else None
    if source_store is not None and not set(frames) <= set(source_store.frames):
        raise ValueError('requested frames are not all present in the source stereo run')
    thresholds = args.thresholds or [config.threshold_px]
    if len(set(thresholds)) != len(thresholds):
        raise ValueError('thresholds must be unique')
    variants = {}
    for threshold in thresholds:
        replace(config, threshold_px=threshold)  # Validate before creating output.
        for version in dict.fromkeys(args.versions):
            name = f'{version}_tau{threshold:g}'
            variants[name] = dict(disparity=f'{name}_disparity_px.npy', version=version, threshold_px=threshold)
            if version == 'v2':
                variants[f'v2_vote_tau{threshold:g}'] = dict(disparity=f'{name}_vote_only_disparity_px.npy',
                    version='v2', ablation='vote_only', threshold_px=threshold)
    metadata = dict(frames=frames, variants=variants, config=asdict(config), thresholds=thresholds,
        model=cache.metadata, mono_cache=str(cache.root), label=args.label or cache.name,
        original=dict(disparity='original_disparity_px.npy'),
        data_root=str(args.data_root.resolve()),
        stereo_root=str(args.stereo_root.resolve()) if args.stereo_root is not None else None,
        stereo_run=str(source_store.root) if source_store is not None else None,
        coordinate_system='input stereo-reference; native RGB mono; disparity px',
        new_neural_inferences=0)
    store = create_store(output, metadata, overwrite=args.overwrite)
    rows = []
    for stem in frames:
        mono, provenance = cache.load(stem, args.data_root/'RGB'/f'{stem}.png')
        original = (source_store.load_original(stem) if source_store is not None else
                    read_stereo_bin(args.stereo_root/f'{stem}_disp.bin', args.height, args.width, scale=64))
        calibration = load_depth_rgb_calibration(args.calibration or args.data_root/'raw_calibration_list.json',
                                                 (args.width, args.height), (mono.shape[1], mono.shape[0]))
        results = {}
        for threshold in thresholds:
            refiner = DepthRefiner(calibration, replace(config, threshold_px=threshold))
            for version, result in refiner.refine_many(original, mono, args.versions).items():
                results[f'{version}_tau{threshold:g}'] = result
        row = export_frame(store, stem, original, results, provenance=provenance)
        rows.append(row)
        print(stem, {k:dict(removed=v['metrics']['removed'], changed=v['metrics']['changed_over_005px'])
                     for k, v in row['variants'].items()}, flush=True)
    finish_store(store, rows)
    print('Gallery:', output/'index.html')


def compare_command(args):
    inputs = [(name, ResultStore(root), variant) for name, root, variant in args.input]
    names = [x[0] for x in inputs]
    if len(set(names)) != len(names) or 'Original' in names:
        raise ValueError('comparison labels must be unique and not Original')
    pairs = []
    for baseline, candidate in args.pair or []:
        if baseline not in names or candidate not in names or baseline == candidate:
            raise ValueError('pair labels must name distinct comparison inputs')
        pairs.append(dict(baseline=baseline, candidate=candidate, retained_again=0,
                          newly_removed=0, maximum_common_change_px=0.))
    if args.crop is not None:
        x0, y0, x1, y1 = args.crop
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError('crop must be a nonempty x0 y0 x1 y1 rectangle')
    frames = args.frames or inputs[0][1].frames
    if not frames or any(not set(frames) <= set(store.frames) for _, store, _ in inputs):
        raise ValueError('comparison requires the same requested frames in every run')
    for _, store, variant in inputs:
        if variant not in store.metadata['variants']:
            raise ValueError(f'unknown variant {variant} in {store.root}')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('comparison output must be empty')
    root.mkdir(parents=True, exist_ok=True)
    totals = {name:dict(removed=0, roughness_sum=0.) for name in ['Original']+names}
    points = stencils = common_points = 0
    audit_totals = {}
    audit_files = []
    for stem in frames:
        original = inputs[0][1].load_original(stem)
        fields = {'Original':original}
        common = valid_disparity(original)
        for name, store, variant in inputs:
            if not np.array_equal(original, store.load_original(stem), equal_nan=True):
                raise ValueError('runs do not have identical original stereo')
            field = store.load(stem, variant)
            if field.shape != original.shape or (valid_disparity(field) & ~valid_disparity(original)).any():
                raise ValueError('shape mismatch or filled original holes')
            fields[name] = field
            common &= valid_disparity(field)
        stencil = five_point_mask(common)
        reference = reliable = None
        if args.audit_cache:
            paths = [path/stem/'dense_evidence.npz' for path in args.audit_cache
                     if (path/stem/'dense_evidence.npz').is_file()]
            if len(paths) != 1:
                raise ValueError(f'expected one audit reference for {stem}, got {len(paths)}')
            with np.load(paths[0], allow_pickle=False) as evidence:
                reference, reliable = evidence['disparity'], evidence['center_checked']
            audit_files.append(dict(stem=stem, path=str(paths[0].resolve()),
                                    sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest()))
            for name, field in fields.items():
                measurements = audit_against_reference(original, field, reference, reliable, common)
                dest = audit_totals.setdefault(name, {})
                for key, value in measurements.items():
                    dest[key] = dest.get(key, 0)+value
        points += int(valid_disparity(original).sum()); common_points += int(common.sum()); stencils += int(stencil.sum())
        for name, field in fields.items():
            totals[name]['removed'] += int((valid_disparity(original) & ~valid_disparity(field)).sum())
            totals[name]['roughness_sum'] += float(np.abs(laplacian(field)[stencil]).astype(float).sum())
        target = root/stem; target.mkdir()
        render_comparison_grid(target/'full_frame_colormap_comparison.jpg', original, fields, pairs)
        if args.crop is not None:
            if x1 > original.shape[1] or y1 > original.shape[0]:
                raise ValueError('crop exceeds input image dimensions')
            render_comparison_grid(target/'edge_comparison.jpg', original, fields, pairs, crop=np.s_[y0:y1, x0:x1])
        for pair_index, pair in enumerate(pairs):
            before, after = fields[pair['baseline']], fields[pair['candidate']]
            kept_before, kept_after = valid_disparity(before), valid_disparity(after)
            restored, removed = ~kept_before & kept_after, kept_before & ~kept_after
            both = kept_before & kept_after
            pair['retained_again'] += int(restored.sum())
            pair['newly_removed'] += int(removed.sum())
            pair['maximum_common_change_px'] = max(pair['maximum_common_change_px'],
                float(np.abs(before[both]-after[both]).max(initial=0)))
            if reference is not None:
                checked = valid_disparity(original) & valid_disparity(reference) & reliable.astype(bool)
                error = np.abs(original-reference)
                categories = dict(supported=checked & (error <= .75), conflicting=checked & (error > 2.),
                                  intermediate=checked & (error > .75) & (error <= 2.), unchecked=~checked)
                for category, mask in categories.items():
                    key = 'retained_again_'+category
                    pair[key] = pair.get(key, 0)+int((restored & mask).sum())
            np.save(target/f'pair_{pair_index}_retained_again_mask.npy', restored)
            np.save(target/f'pair_{pair_index}_newly_removed_mask.npy', removed)
            image = (colorize_disparity(original, 10, 192)*.3).astype(np.uint8)
            image[restored] = (0, 255, 0); image[removed] = (0, 0, 255)
            if not cv2.imwrite(str(target/f'pair_{pair_index}_change_map.png'), image):
                raise OSError('failed to write validity-change visualization')
    for value in totals.values():
        value['removed_fraction'] = value['removed']/max(points, 1)
        value['common_mean_abs_laplacian_px'] = value['roughness_sum']/stencils if stencils else None
    summary = dict(frames=frames, original_valid=points, common_retained=common_points,
        common_stencils=stencils, methods=totals,
        inputs=[dict(label=name, root=str(store.root), variant=variant) for name, store, variant in inputs],
        pairs=pairs, audit=audit_totals, audit_files=audit_files,
        warning='Only common retained stencils are compared. Roughness and cached-reference agreement are not physical accuracy.')
    write_json(root/'aggregate.json', summary)
    write_comparison_report(root, summary)
    write_gallery(root, frames, 'V1/V2 run comparison',
                  'Shared 10–192px scale. Black = invalid. Labels identify runs, not confidence.',
                  change_labels=[f"{p['baseline']} -> {p['candidate']}" for p in pairs])
    print('Gallery:', root/'index.html')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='refine using existing native RGB mono caches')
    run.add_argument('--data-root', type=Path, required=True)
    stereo = run.add_mutually_exclusive_group(required=True)
    stereo.add_argument('--stereo-root', type=Path, help='normally oriented raw stereo directory')
    stereo.add_argument('--stereo-run', type=Path, help='read original stereo from an existing result manifest')
    run.add_argument('--mono-cache', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument('--frames', nargs='+')
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--sample20', action='store_true')
    run.add_argument('--versions', nargs='+', choices=['v1', 'v2'], default=['v1', 'v2'])
    run.add_argument('--thresholds', type=float, nargs='+')
    run.add_argument('--config', type=Path)
    run.add_argument('--calibration', type=Path)
    run.add_argument('--width', type=int, default=1280)
    run.add_argument('--height', type=int, default=720)
    run.add_argument('--label')
    run.add_argument('--allow-legacy-unhashed-cache', action='store_true')
    run.add_argument('--overwrite', action='store_true')
    run.set_defaults(handler=run_command)
    compare = sub.add_parser('compare', help='compare named variants from any V1/V2 result stores')
    compare.add_argument('--input', nargs=3, action='append', required=True, metavar=('LABEL', 'RUN_ROOT', 'VARIANT'))
    compare.add_argument('--output', type=Path, required=True)
    compare.add_argument('--frames', nargs='+')
    compare.add_argument('--crop', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'))
    compare.add_argument('--pair', nargs=2, action='append', metavar=('BASELINE', 'CANDIDATE'))
    compare.add_argument('--audit-cache', type=Path, action='append',
                         help='optional diagnostic-only ROOT/<frame>/dense_evidence.npz')
    compare.set_defaults(handler=compare_command)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
