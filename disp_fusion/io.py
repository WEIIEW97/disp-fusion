from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_depth_png(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def read_stereo_bin(
    path: Path,
    height: int,
    width: int,
    scale: float = 64.0,
    undo_hflip: bool = False,
) -> np.ndarray:
    """Decode disparity in its stored reference coordinates by default.

    ``undo_hflip`` is an explicit legacy import option, not a refinement step.
    Normal inputs and the CLI preserve the stored image orientation.
    """
    raw = np.fromfile(path, dtype=np.int16)
    expected = height * width
    if raw.size != expected:
        raise ValueError(f"{path}: got {raw.size} int16 values, expected {expected}")
    disparity = raw.reshape(height, width).astype(np.float32)
    disparity[disparity < 0] = np.nan
    disparity /= scale
    if undo_hflip:
        disparity = disparity[:, ::-1].copy()
    return disparity


def write_q6(path, disparity):
    """Export input-reference disparity; never silently clip physical values."""
    from .alignment import valid_disparity
    valid = valid_disparity(disparity)
    values = np.rint(disparity[valid]*64)
    if np.any(values > 32767) or np.any(values < 1):
        raise ValueError('valid disparity is outside positive int16 Q6 range')
    result = np.full(disparity.shape, -1, np.int16)
    result[valid] = values.astype(np.int16)
    result.tofile(path)
    return result
