"""Shared colormaps, grids and static galleries, independent of algorithm."""
from pathlib import Path
import html
import cv2
import numpy as np


def colorize_disparity(disparity: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    normalized = np.zeros(disparity.shape, np.uint8)
    if valid.any():
        if lo is None or hi is None:
            lo0, hi0 = np.percentile(disparity[valid], [2.0, 98.0])
            lo = float(lo0) if lo is None else lo
            hi = float(hi0) if hi is None else hi
        normalized[valid] = np.clip(
            (disparity[valid] - lo) / max(float(hi) - float(lo), 1e-6) * 255.0,
            0,
            255,
        ).astype(np.uint8)
    result = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def colorize_relative_prediction(prediction: np.ndarray) -> np.ndarray:
    """Relative mono output: finite zero is far depth, not an invalid hole."""
    valid=np.isfinite(prediction)
    normalized=np.zeros(prediction.shape,np.uint8)
    if valid.any():
        lo,hi=np.percentile(prediction[valid],[2,98])
        normalized[valid]=np.clip((prediction[valid]-lo)/max(float(hi-lo),1e-6)*255,0,255).astype(np.uint8)
    image=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
    image[~valid]=0
    return image


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def panel(image, title, width=640, height=380):
    scale = min(width/image.shape[1], (height-34)/image.shape[0])
    w, h = round(image.shape[1]*scale), round(image.shape[0]*scale)
    out = np.zeros((height, width, 3), np.uint8)
    x, y = (width-w)//2, 34+(height-34-h)//2
    out[y:y+h, x:x+w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    return add_label(out, title)


def render_grid(path, fields, *, crop=None, columns=2, scale=(10, 192), native_resolution=True):
    if not fields or columns < 1:
        raise ValueError('a grid needs fields and positive column count')
    width, height = (1280, 754) if native_resolution and crop is None else (640, 380)
    tiles = [panel(colorize_disparity(field if crop is None else field[crop], *scale),
                   f'{name} / {scale[0]}..{scale[1]}px', width, height)
             for name, field in fields.items()]
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k+columns]) for k in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f'could not write {path}')


def render_comparison_grid(path, original, fields, pairs=(), *, crop=None):
    """Keep each requested baseline/candidate pair on one row with its input."""
    if pairs:
        arranged = {}
        used = set()
        for index, pair in enumerate(pairs):
            arranged[f'Original [pair {index+1}]'] = original
            for name in [pair['baseline'], pair['candidate']]:
                arranged[f'{name} [pair {index+1}]'] = fields[name]
                used.add(name)
        for name, field in fields.items():
            if name != 'Original' and name not in used:
                arranged[name] = field
    else:
        arranged = fields
    render_grid(path, arranged, crop=crop, columns=3 if pairs else 2,
                scale=(20, 150) if crop is not None else (10, 192))


def write_gallery(root, frames, title, description, *, change_labels=()):
    root = Path(root)
    parts = ['<!doctype html><meta charset="utf-8">',
             '<style>body{background:#151515;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%}a{color:#8cf}</style>',
             '<h1>'+html.escape(title)+'</h1>', '<p>'+html.escape(description)+'</p>',
             '<a href="aggregate.json">Statistics JSON</a>']
    for stem in frames:
        parts += ['<h2>'+html.escape(stem)+'</h2>',
                  f'<img loading="lazy" src="{stem}/full_frame_colormap_comparison.jpg">']
        if (root/stem/'edge_comparison.jpg').exists():
            parts += [f'<img loading="lazy" src="{stem}/edge_comparison.jpg">']
        for path in sorted((root/stem).glob('pair_*_change_map.png')):
            index = int(path.stem.split('_')[1])
            label = change_labels[index] if index < len(change_labels) else path.stem
            parts += ['<details><summary>'+html.escape(label)+': green = retained again, red = newly removed; not correctness labels</summary>',
                      f'<img loading="lazy" src="{stem}/{path.name}"></details>']
    if (root/'REPORT.md').exists():
        parts += ['<p><a href="REPORT.md">Comparison report (English)</a></p>']
    (root/'index.html').write_text('\n'.join(parts))
