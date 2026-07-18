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
        mapped = cv2.perspectiveTransform(
            camera_points.reshape(-1, 1, 2),
            np.asarray(calibration.camera_to_field),
        ).reshape(-1, 2)
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


if __name__ == "__main__":
    unittest.main()
