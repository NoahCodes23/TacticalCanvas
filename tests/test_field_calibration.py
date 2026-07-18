import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tactical_canvas.calibration.layout import (
    REQUIRED_MARKER_IDS,
    create_field_marker_layout,
)
from tactical_canvas.calibration.models import FieldCalibration, Size
from tactical_canvas.calibration.solver import CalibrationAccumulator


class FieldCalibrationTests(unittest.TestCase):
    def test_layout_stays_inside_field_and_renders_square(self):
        layout = create_field_marker_layout()
        self.assertEqual([marker.marker_id for marker in layout], list(range(20, 26)))
        for marker in layout:
            self.assertGreaterEqual(marker.x - marker.quiet_x, 0)
            self.assertGreaterEqual(marker.y - marker.quiet_y, 0)
            self.assertLessEqual(marker.x + marker.width + marker.quiet_x, 1)
            self.assertLessEqual(marker.y + marker.height + marker.quiet_y, 1)
            self.assertAlmostEqual(marker.width * 105, marker.height * 68)

    def test_solver_maps_camera_pixels_directly_to_field(self):
        layout = create_field_marker_layout()
        field_to_camera = np.asarray(
            [[540.0, 35.0, 80.0], [20.0, 360.0, 55.0], [0.05, 0.03, 1.0]],
            dtype=np.float64,
        )
        accumulator = CalibrationAccumulator(layout, minimum_samples=3)
        expected = {marker.marker_id: marker.corners for marker in layout}
        detected = {
            marker_id: cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), field_to_camera
            ).reshape(4, 2)
            for marker_id, corners in expected.items()
            if marker_id in REQUIRED_MARKER_IDS
        }
        for _ in range(3):
            accumulator.add_frame(detected)

        calibration = accumulator.solve_field(Size(640, 480), camera_index=1)
        camera_points = np.concatenate(list(detected.values()))
        mapped = calibration.camera_points_to_field(camera_points)
        field_points = np.concatenate(
            [expected[marker_id] for marker_id in REQUIRED_MARKER_IDS]
        )
        np.testing.assert_allclose(mapped, field_points, atol=1e-5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.json"
            calibration.save(path)
            loaded = FieldCalibration.load(path)
        self.assertEqual(loaded.camera_size, Size(640, 480))
        self.assertEqual(loaded.markers_used, list(REQUIRED_MARKER_IDS))
        self.assertEqual(np.asarray(loaded.correction_coefficients).shape, (2, 10))

    def test_residual_warp_reduces_smooth_lens_error(self):
        layout = create_field_marker_layout()
        field_to_camera = np.asarray(
            [[540.0, 35.0, 80.0], [20.0, 360.0, 55.0], [0.05, 0.03, 1.0]],
            dtype=np.float64,
        )
        expected = {marker.marker_id: marker.corners for marker in layout}

        def distort(points):
            values = points.astype(np.float64).copy()
            dx = (values[:, 0] - 320.0) / 320.0
            dy = (values[:, 1] - 240.0) / 320.0
            factor = 1.0 - 0.10 * (dx * dx + dy * dy)
            values[:, 0] = 320.0 + dx * factor * 320.0
            values[:, 1] = 240.0 + dy * factor * 320.0
            return values.astype(np.float32)

        detected = {}
        for marker_id, corners in expected.items():
            projected = cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), field_to_camera
            ).reshape(4, 2)
            detected[marker_id] = distort(projected)

        accumulator = CalibrationAccumulator(layout, minimum_samples=3)
        for _ in range(3):
            accumulator.add_frame(detected)
        calibration = accumulator.solve_field(Size(640, 480), camera_index=1)

        camera_points = np.concatenate(
            [detected[marker.marker_id] for marker in layout]
        )
        field_points = np.concatenate([marker.corners for marker in layout])
        base = cv2.perspectiveTransform(
            camera_points.reshape(-1, 1, 2),
            np.asarray(calibration.camera_to_field),
        ).reshape(-1, 2)
        corrected = calibration.correct_field_points(base)
        base_rmse = np.sqrt(np.mean(np.square(base - field_points)))
        corrected_rmse = np.sqrt(np.mean(np.square(corrected - field_points)))
        self.assertLess(corrected_rmse, base_rmse * 0.6)


if __name__ == "__main__":
    unittest.main()
