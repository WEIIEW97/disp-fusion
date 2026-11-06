from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class DepthRgbCalibration:
    depth: CameraIntrinsics
    rgb: CameraIntrinsics
    rgb_distortion: np.ndarray
    depth_to_rgb_rotation: np.ndarray
    depth_to_rgb_translation_mm: np.ndarray
    baseline_mm: float
    parameter_index: int

    @property
    def disparity_scale_px_mm(self) -> float:
        return self.depth.fx * self.baseline_mm


@dataclass
class DepthToRgbProjection:
    equivalent_disparity: np.ndarray
    rgb_z_mm: np.ndarray
    source_flat_index: np.ndarray
    projected_u: np.ndarray
    projected_v: np.ndarray


def _intrinsics(payload: dict[str, float | int]) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
    )


def load_depth_rgb_calibration(
    path: Path | str,
    depth_size: tuple[int, int],
    rgb_size: tuple[int, int],
) -> DepthRgbCalibration:
    """Load the unique calibration entry matching (width, height) pairs."""

    payload = json.loads(Path(path).read_text())
    matches = []
    for parameter in payload["params"]:
        depth = parameter["depth_intrinsic"]
        rgb = parameter["rgb_intrinsic"]
        if (
            int(depth["width"]),
            int(depth["height"]),
        ) == depth_size and (int(rgb["width"]), int(rgb["height"])) == rgb_size:
            matches.append(parameter)
    if len(matches) != 1:
        indices = [item.get("index") for item in matches]
        raise ValueError(
            f"expected one calibration for depth={depth_size}, rgb={rgb_size}; "
            f"found indices {indices}"
        )
    parameter = matches[0]
    distortion = parameter["rgb_distortion"]
    coefficients = np.asarray(
        [
            distortion["k1"],
            distortion["k2"],
            distortion["p1"],
            distortion["p2"],
            distortion["k3"],
            distortion["k4"],
            distortion["k5"],
            distortion["k6"],
        ],
        dtype=np.float64,
    )
    transform = parameter["transform"]
    return DepthRgbCalibration(
        depth=_intrinsics(parameter["depth_intrinsic"]),
        rgb=_intrinsics(parameter["rgb_intrinsic"]),
        rgb_distortion=coefficients,
        depth_to_rgb_rotation=np.asarray(transform["rot"], dtype=np.float64),
        depth_to_rgb_translation_mm=np.asarray(transform["trans"], dtype=np.float64),
        baseline_mm=float(payload["baseline_mm"]),
        parameter_index=int(parameter["index"]),
    )


def _distort_points(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    radius2 = x * x + y * y
    radius4 = radius2 * radius2
    radius6 = radius4 * radius2
    radial = (1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6) / (
        1.0 + k4 * radius2 + k5 * radius4 + k6 * radius6
    )
    distorted_x = (
        x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    )
    distorted_y = (
        y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    )
    return distorted_x, distorted_y


def project_depth_disparity_to_rgb(
    disparity: np.ndarray,
    calibration: DepthRgbCalibration,
) -> DepthToRgbProjection:
    """Z-buffer a depth-camera disparity image into the distorted RGB view."""

    expected = (calibration.depth.height, calibration.depth.width)
    if disparity.shape != expected:
        raise ValueError(f"depth disparity shape {disparity.shape} != {expected}")
    valid = np.isfinite(disparity) & (disparity > 0.05)
    yy, xx = np.indices(disparity.shape, dtype=np.float64)
    source_index = np.flatnonzero(valid)
    z_depth = calibration.disparity_scale_px_mm / disparity[valid].astype(np.float64)
    points_depth = np.column_stack(
        [
            (xx[valid] - calibration.depth.cx) / calibration.depth.fx * z_depth,
            (yy[valid] - calibration.depth.cy) / calibration.depth.fy * z_depth,
            z_depth,
        ]
    )
    points_rgb = (
        points_depth @ calibration.depth_to_rgb_rotation.T
        + calibration.depth_to_rgb_translation_mm
    )
    z_rgb = points_rgb[:, 2]
    x_rgb = points_rgb[:, 0] / z_rgb
    y_rgb = points_rgb[:, 1] / z_rgb
    x_rgb, y_rgb = _distort_points(x_rgb, y_rgb, calibration.rgb_distortion)
    projected_u = calibration.rgb.fx * x_rgb + calibration.rgb.cx
    projected_v = calibration.rgb.fy * y_rgb + calibration.rgb.cy
    integer_u = np.rint(projected_u).astype(np.int32)
    integer_v = np.rint(projected_v).astype(np.int32)
    inside = (
        (integer_u >= 0)
        & (integer_u < calibration.rgb.width)
        & (integer_v >= 0)
        & (integer_v < calibration.rgb.height)
        & (z_rgb > 0.0)
    )
    target_index = integer_v[inside] * calibration.rgb.width + integer_u[inside]
    target_z = z_rgb[inside]
    target_source = source_index[inside]
    # Lexicographic sorting makes the nearest point the first item for every
    # target pixel, implementing a deterministic z-buffer.
    order = np.lexsort((target_z, target_index))
    target_index = target_index[order]
    target_z = target_z[order]
    target_source = target_source[order]
    first = np.r_[True, target_index[1:] != target_index[:-1]] if len(target_index) else np.zeros(0, bool)
    target_index = target_index[first]
    target_z = target_z[first]
    target_source = target_source[first]

    rgb_shape = (calibration.rgb.height, calibration.rgb.width)
    rgb_z = np.full(np.prod(rgb_shape), np.nan, dtype=np.float32)
    rgb_source = np.full(np.prod(rgb_shape), -1, dtype=np.int64)
    rgb_z[target_index] = target_z.astype(np.float32)
    rgb_source[target_index] = target_source
    rgb_z = rgb_z.reshape(rgb_shape)
    equivalent = np.full(rgb_shape, np.nan, dtype=np.float32)
    rgb_valid = np.isfinite(rgb_z) & (rgb_z > 0.0)
    equivalent[rgb_valid] = calibration.disparity_scale_px_mm / rgb_z[rgb_valid]
    return DepthToRgbProjection(
        equivalent_disparity=equivalent,
        rgb_z_mm=rgb_z,
        source_flat_index=rgb_source.reshape(rgb_shape),
        projected_u=projected_u.astype(np.float32),
        projected_v=projected_v.astype(np.float32),
    )
