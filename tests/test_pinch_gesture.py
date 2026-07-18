import unittest
from types import SimpleNamespace

from server.state import AppState
from vision.gestures import HandTracker, pinch_pointer


def landmarks(
    thumb_x: float,
    index_x: float,
    *,
    middle_x: float = 0.62,
    ring_x: float = 0.69,
    pinky_x: float = 0.76,
):
    points = [SimpleNamespace(x=0.5, y=0.6, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    points[5] = SimpleNamespace(x=0.43, y=0.52, z=0.0)
    points[9] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    points[17] = SimpleNamespace(x=0.66, y=0.55, z=0.0)
    points[4] = SimpleNamespace(x=thumb_x, y=0.3, z=0.0)
    points[8] = SimpleNamespace(x=index_x, y=0.3, z=0.0)
    points[12] = SimpleNamespace(x=middle_x, y=0.3, z=0.0)
    points[16] = SimpleNamespace(x=ring_x, y=0.3, z=0.0)
    points[20] = SimpleNamespace(x=pinky_x, y=0.3, z=0.0)
    return points


class PinchGestureTests(unittest.TestCase):
    def test_pointer_is_midpoint_of_thumb_and_index_tips(self):
        pointer, ratio, index_primary = pinch_pointer(
            landmarks(0.30, 0.50), 200, 100
        )
        self.assertEqual(pointer, (80.0, 30.0))
        self.assertGreater(ratio, 0.58)
        self.assertTrue(index_primary)

        pinched_pointer, pinched_ratio, index_primary = pinch_pointer(
            landmarks(0.39, 0.41), 200, 100
        )
        self.assertEqual(pinched_pointer, (80.0, 30.0))
        self.assertLess(pinched_ratio, ratio)
        self.assertTrue(index_primary)

    def test_world_landmarks_make_pinch_independent_of_image_scale(self):
        near_image = landmarks(0.30, 0.50)
        far_image = landmarks(0.47, 0.53)
        world = landmarks(0.39, 0.41)

        _, near_ratio, _ = pinch_pointer(near_image, 200, 100, world)
        _, far_ratio, _ = pinch_pointer(far_image, 200, 100, world)

        self.assertAlmostEqual(near_ratio, far_ratio)

    def test_middle_finger_contact_does_not_start_a_grab(self):
        tracker = HandTracker("Right")
        middle_pinch = landmarks(
            0.40, 0.48, middle_x=0.405, ring_x=0.60, pinky_x=0.70
        )
        _, ratio, index_primary = pinch_pointer(middle_pinch, 200, 100)

        self.assertFalse(index_primary)
        self.assertEqual(
            tracker.update(
                (0.5, 0.5), ratio, 1.00, index_is_primary=index_primary
            ),
            "hover",
        )
        self.assertEqual(
            tracker.update(
                (0.5, 0.5), ratio, 1.01, index_is_primary=index_primary
            ),
            "hover",
        )
        self.assertFalse(tracker.grabbing)

    def test_close_pinch_grabs_and_open_pinch_releases(self):
        tracker = HandTracker("Right")
        self.assertEqual(tracker.update((0.5, 0.5), 0.2, 1.00), "hover")
        self.assertEqual(tracker.update((0.5, 0.5), 0.2, 1.01), "grab_start")
        self.assertEqual(tracker.update((0.5, 0.5), 0.45, 1.02), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.03), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.04), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.05), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.06), "grab_end")

    def test_one_open_pinch_glitch_does_not_drop_a_piece(self):
        tracker = HandTracker("Right")
        tracker.update((0.5, 0.5), 0.2, 1.00)
        self.assertEqual(tracker.update((0.5, 0.5), 0.2, 1.01), "grab_start")
        self.assertEqual(tracker.update((0.5, 0.5), 0.8, 1.02), "grab_move")
        self.assertEqual(tracker.update((0.5, 0.5), 0.3, 1.03), "grab_move")
        self.assertTrue(tracker.grabbing)

    def test_grab_snaps_to_a_player_one_piece_width_away(self):
        state = AppState.__new__(AppState)
        state.players = [SimpleNamespace(id="H7", x=10.0, y=10.0)]
        state.grabbed = {}
        self.assertEqual(state.nearest_player(13.9, 10.0).id, "H7")
        self.assertIsNone(state.nearest_player(14.1, 10.0))


if __name__ == "__main__":
    unittest.main()
