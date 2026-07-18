import unittest
from types import SimpleNamespace

from vision.gestures import HandTracker, pinch_pointer


def landmarks(thumb_x: float, index_x: float):
    points = [SimpleNamespace(x=0.5, y=0.6, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    points[9] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    points[4] = SimpleNamespace(x=thumb_x, y=0.3, z=0.0)
    points[8] = SimpleNamespace(x=index_x, y=0.3, z=0.0)
    return points


class PinchGestureTests(unittest.TestCase):
    def test_pointer_is_midpoint_of_thumb_and_index_tips(self):
        pointer, ratio = pinch_pointer(landmarks(0.30, 0.50), 200, 100)
        self.assertEqual(pointer, (80.0, 30.0))
        self.assertGreater(ratio, 1.0)

    def test_close_pinch_grabs_and_open_pinch_releases(self):
        tracker = HandTracker("Right")
        self.assertEqual(tracker.update((0.5, 0.5), 0.2, 1.00), "hover")
        self.assertEqual(tracker.update((0.5, 0.5), 0.2, 1.01), "grab_start")
        self.assertEqual(tracker.update((0.5, 0.5), 0.45, 1.02), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.03), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.04), "grab_end")


if __name__ == "__main__":
    unittest.main()
