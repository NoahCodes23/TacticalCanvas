from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .layout import FieldMarkerPlacement, REQUIRED_MARKER_IDS, MarkerPlacement
from .models import (
    DisplayInfo,
    FieldCalibration,
    ProjectorCalibration,
    Size,
    field_warp_basis,
)


class CalibrationAccumulator:
    """Collects repeated marker observations and solves a planar homography."""

    def __init__(
        self,
        layout: list[MarkerPlacement] | list[FieldMarkerPlacement],
        minimum_samples: int = 14,
        maximum_samples: int = 60,
        required_marker_ids: tuple[int, ...] = REQUIRED_MARKER_IDS,
    ) -> None:
        self._layout = {placement.marker_id: placement for placement in layout}
        self.minimum_samples = minimum_samples
        self.maximum_samples = maximum_samples
        self.required_marker_ids = required_marker_ids
        self._samples: dict[int, list[np.ndarray]] = defaultdict(list)

    def add_frame(self, detected: dict[int, np.ndarray]) -> None:
        for marker_id, corners in detected.items():
            if marker_id not in self._layout:
                continue
            value = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            samples = self._samples[marker_id]
            samples.append(value.copy())
            if len(samples) > self.maximum_samples:
                del samples[0]

    def sample_count(self, marker_id: int) -> int:
        return len(self._samples.get(marker_id, ()))

    @property
    def ready(self) -> bool:
        return all(
            self.sample_count(marker_id) >= self.minimum_samples
            for marker_id in self.required_marker_ids
        )

    @property
    def progress(self) -> float:
        least_samples = min(
            self.sample_count(marker_id) for marker_id in self.required_marker_ids
        )
        return min(1.0, least_samples / self.minimum_samples)

    @property
    def required_visible(self) -> int:
        return sum(
            self.sample_count(marker_id) > 0
            for marker_id in self.required_marker_ids
        )

    def solve(
        self,
        camera_size: Size,
        projector_size: Size,
        display: DisplayInfo | None = None,
        camera_index: int = 0,
        camera_fps: float = 30.0,
    ) -> ProjectorCalibration:
        if not self.ready:
            raise RuntimeError("All required markers need more observations")

        camera_points: list[np.ndarray] = []
        projector_points: list[np.ndarray] = []
        jitter_distances: list[np.ndarray] = []
        markers_used: list[int] = []

        for marker_id in sorted(self._samples):
            samples = self._samples[marker_id]
            if len(samples) < self.minimum_samples:
                continue
            representative = np.median(np.stack(samples), axis=0).astype(np.float32)
            camera_points.append(representative)
            projector_points.append(self._layout[marker_id].corners)
            jitter_distances.extend(np.linalg.norm(sample - representative, axis=1) for sample in samples)
            markers_used.append(marker_id)

        source = np.concatenate(camera_points).astype(np.float32)
        destination = np.concatenate(projector_points).astype(np.float32)
        camera_to_projector, _mask = cv2.findHomography(source, destination, method=0)
        if camera_to_projector is None or not np.isfinite(camera_to_projector).all():
            raise RuntimeError("OpenCV could not solve the calibration homography")
        projector_to_camera = np.linalg.inv(camera_to_projector)
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), camera_to_projector).reshape(-1, 2)
        errors = np.linalg.norm(projected - destination, axis=1)
        all_jitter = np.concatenate(jitter_distances)

        return ProjectorCalibration(
            camera_size=camera_size,
            projector_size=projector_size,
            display=display
            or DisplayInfo(index=0, x=0, y=0, width=projector_size.width, height=projector_size.height),
            camera_index=camera_index,
            camera_fps=camera_fps,
            camera_to_projector=camera_to_projector.tolist(),
            projector_to_camera=projector_to_camera.tolist(),
            reprojection_rmse=float(np.sqrt(np.mean(np.square(errors)))),
            camera_jitter=float(np.sqrt(np.mean(np.square(all_jitter)))),
            markers_used=markers_used,
        )

    def solve_field(
        self,
        camera_size: Size,
        camera_index: int = 0,
        camera_fps: float = 30.0,
    ) -> FieldCalibration:
        """Solve camera pixels directly into normalized field coordinates."""

        source, destination, all_jitter, markers_used = self._correspondences()
        camera_to_field, _mask = cv2.findHomography(source, destination, method=0)
        if camera_to_field is None or not np.isfinite(camera_to_field).all():
            raise RuntimeError("OpenCV could not solve the field calibration homography")
        try:
            field_to_camera = np.linalg.inv(camera_to_field)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("The field calibration homography is singular") from error
        projected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), camera_to_field
        ).reshape(-1, 2)
        basis = field_warp_basis(projected)
        # Perspective is handled by the homography. The remaining smooth error
        # across the flat field is predominantly camera/projector lens curvature.
        regularization = 1e-5
        penalty = np.eye(basis.shape[1]) * regularization
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            basis.T @ basis + penalty,
            basis.T @ (destination - projected),
        )
        corrected = projected + basis @ coefficients
        errors = np.linalg.norm(corrected - destination, axis=1)
        return FieldCalibration(
            camera_size=camera_size,
            camera_index=camera_index,
            camera_fps=camera_fps,
            camera_to_field=camera_to_field.tolist(),
            field_to_camera=field_to_camera.tolist(),
            reprojection_rmse=float(np.sqrt(np.mean(np.square(errors)))),
            camera_jitter=float(np.sqrt(np.mean(np.square(all_jitter)))),
            markers_used=markers_used,
            correction_coefficients=coefficients.T.tolist(),
        )

    def _correspondences(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        if not self.ready:
            raise RuntimeError("All required markers need more observations")

        camera_points: list[np.ndarray] = []
        destination_points: list[np.ndarray] = []
        jitter_distances: list[np.ndarray] = []
        markers_used: list[int] = []
        for marker_id in sorted(self._samples):
            samples = self._samples[marker_id]
            if len(samples) < self.minimum_samples:
                continue
            representative = np.median(np.stack(samples), axis=0).astype(np.float32)
            camera_points.append(representative)
            destination_points.append(self._layout[marker_id].corners)
            jitter_distances.extend(
                np.linalg.norm(sample - representative, axis=1) for sample in samples
            )
            markers_used.append(marker_id)

        return (
            np.concatenate(camera_points).astype(np.float32),
            np.concatenate(destination_points).astype(np.float32),
            np.concatenate(jitter_distances),
            markers_used,
        )
