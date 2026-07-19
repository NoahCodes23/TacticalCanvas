import math
import unittest
from types import SimpleNamespace

from server.state import AppState
from vision.gestures import (
    HandTracker,
    drawing_pointer,
    paint_gestures,
    pinch_pointer,
)


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


def drawing_landmarks(
    middle_tip_x: float = 0.50,
    *,
    middle_extended: bool = False,
    ring_extended: bool = False,
    five_extended: bool = False,
):
    points = [SimpleNamespace(x=0.5, y=0.6, z=0.0) for _ in range(21)]
    coordinates = {
        0: (0.50, 0.80),
        2: (0.44, 0.64), 3: (0.43, 0.61), 4: (0.44, 0.60),
        5: (0.42, 0.55), 6: (0.43, 0.45),
        7: (0.44, 0.35), 8: (0.45, 0.25),
        9: (0.50, 0.54), 10: (0.50, 0.44),
        11: (0.50, 0.50), 12: (middle_tip_x, 0.56),
        13: (0.57, 0.56), 14: (0.58, 0.50),
        15: (0.59, 0.54), 16: (0.60, 0.58),
        17: (0.65, 0.58), 18: (0.66, 0.52),
        19: (0.67, 0.56), 20: (0.67, 0.60),
    }
    if middle_extended or ring_extended or five_extended:
        coordinates.update({
            9: (0.50, 0.54), 10: (0.50, 0.44),
            11: (0.50, 0.34), 12: (middle_tip_x, 0.25),
        })
    if ring_extended or five_extended:
        coordinates.update({
            13: (0.56, 0.55), 14: (0.56, 0.45),
            15: (0.56, 0.35), 16: (0.55, 0.25),
        })
    if five_extended:
        coordinates.update({
            2: (0.40, 0.62), 3: (0.32, 0.56), 4: (0.22, 0.50),
            17: (0.65, 0.58), 18: (0.66, 0.48),
            19: (0.65, 0.37), 20: (0.63, 0.26),
        })
    for index, (x, y) in coordinates.items():
        points[index] = SimpleNamespace(x=x, y=y, z=0.0)
    return points


class PinchGestureTests(unittest.TestCase):
    def test_pointer_is_midpoint_of_thumb_and_index_tips(self):
        pointer, ratio, world_ratio = pinch_pointer(
            landmarks(0.30, 0.50), 200, 100
        )
        self.assertEqual(pointer, (80.0, 30.0))
        self.assertGreater(ratio, 0.72)
        self.assertIsNone(world_ratio)

        pinched_pointer, pinched_ratio, world_ratio = pinch_pointer(
            landmarks(0.39, 0.41), 200, 100
        )
        self.assertEqual(pinched_pointer, (80.0, 30.0))
        self.assertLess(pinched_ratio, ratio)
        self.assertIsNone(world_ratio)

    def test_visibly_closed_pinch_wins_when_world_estimate_is_open(self):
        image = landmarks(
            0.39, 0.41, middle_x=0.405, ring_x=0.60, pinky_x=0.70
        )
        world = landmarks(0.30, 0.50)
        _, image_ratio, world_ratio = pinch_pointer(image, 200, 100, world)
        tracker = HandTracker("Right")

        self.assertLess(image_ratio, 0.45)
        self.assertGreater(world_ratio, 0.58)
        self.assertEqual(
            tracker.update((0.5, 0.5), image_ratio, 1.00, world_ratio),
            "hover",
        )
        self.assertEqual(
            tracker.update((0.5, 0.5), image_ratio, 1.01, world_ratio),
            "grab_start",
        )

    def test_world_estimate_only_resolves_an_ambiguous_visible_pinch(self):
        image = landmarks(0.36, 0.47)
        world = landmarks(0.39, 0.41)
        _, image_ratio, world_ratio = pinch_pointer(image, 200, 100, world)
        tracker = HandTracker("Right")

        self.assertGreater(image_ratio, 0.45)
        self.assertLess(image_ratio, 0.72)
        self.assertLess(world_ratio, 0.38)
        tracker.update((0.5, 0.5), image_ratio, 1.00, world_ratio)
        self.assertEqual(
            tracker.update((0.5, 0.5), image_ratio, 1.01, world_ratio),
            "grab_start",
        )

    def test_visibly_open_pinch_releases_when_world_estimate_is_closed(self):
        tracker = HandTracker("Right")
        tracker.update((0.5, 0.5), 0.2, 1.00, 0.2)
        tracker.update((0.5, 0.5), 0.2, 1.01, 0.2)

        for frame in range(3):
            self.assertEqual(
                tracker.update((0.5, 0.5), 0.9, 1.02 + frame / 100, 0.2),
                "grab_move",
            )
        self.assertEqual(
            tracker.update((0.5, 0.5), 0.9, 1.05, 0.2), "grab_end"
        )

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

    def test_single_extended_index_finger_forms_drawing_pose(self):
        pointer, active = drawing_pointer(drawing_landmarks(), 200, 100)
        self.assertEqual(pointer, (90.0, 25.0))
        self.assertTrue(active)

        _, two_fingers = drawing_pointer(
            drawing_landmarks(middle_extended=True), 200, 100
        )
        self.assertFalse(two_fingers)

    def test_palm_direction_does_not_block_index_finger_drawing(self):
        image = drawing_landmarks()
        contradictory_world = drawing_landmarks()
        contradictory_world[8] = SimpleNamespace(x=0.1, y=0.1, z=0.5)
        contradictory_world[12] = SimpleNamespace(x=0.9, y=0.9, z=-0.5)

        _, active = drawing_pointer(
            image, 200, 100, contradictory_world
        )
        self.assertTrue(active)

    def test_two_fingers_erase_and_extra_fingers_activate_no_tool(self):
        _, erase_pointer, draw, erase = paint_gestures(
            drawing_landmarks(middle_extended=True), 200, 100
        )
        self.assertEqual(erase_pointer, (95.0, 25.0))
        self.assertFalse(draw)
        self.assertTrue(erase)

        _, _, draw, erase = paint_gestures(
            drawing_landmarks(ring_extended=True), 200, 100
        )
        self.assertFalse(draw)
        self.assertFalse(erase)

    def test_drawing_pose_is_debounced_at_both_ends(self):
        tracker = HandTracker("Right")
        self.assertEqual(
            tracker.update(
                (0.5, 0.5), 0.9, 1.00,
                draw_pose=True, paint_mode=True,
            ),
            "hover",
        )
        self.assertEqual(
            tracker.update(
                (0.5, 0.5), 0.9, 1.01,
                draw_pose=True, paint_mode=True,
            ),
            "draw_start",
        )
        self.assertEqual(
            tracker.update(
                (0.51, 0.5), 0.9, 1.02,
                draw_pose=True, paint_mode=True,
            ),
            "draw_move",
        )
        for frame in range(2):
            self.assertEqual(
                tracker.update(
                    (0.52, 0.5), 0.9, 1.03 + frame / 100,
                    draw_pose=False, paint_mode=True,
                ),
                "draw_move",
            )
        self.assertEqual(
            tracker.update(
                (0.52, 0.5), 0.9, 1.05,
                draw_pose=False, paint_mode=True,
            ),
            "draw_end",
        )

    def test_eraser_gesture_is_debounced_at_both_ends(self):
        tracker = HandTracker("Right")
        self.assertEqual(
            tracker.update(
                (0.5, 0.5), 0.9, 1.00,
                erase_pose=True, paint_mode=True,
            ),
            "hover",
        )
        self.assertEqual(
            tracker.update(
                (0.5, 0.5), 0.9, 1.01,
                erase_pose=True, paint_mode=True,
            ),
            "erase_start",
        )
        self.assertEqual(
            tracker.update(
                (0.51, 0.5), 0.9, 1.02,
                erase_pose=True, paint_mode=True,
            ),
            "erase_move",
        )
        for frame in range(2):
            self.assertEqual(
                tracker.update(
                    (0.51, 0.5), 0.9, 1.03 + frame / 100,
                    paint_mode=True,
                ),
                "erase_move",
            )
        self.assertEqual(
            tracker.update(
                (0.51, 0.5), 0.9, 1.05, paint_mode=True
            ), "erase_end"
        )

    def test_paint_mode_never_falls_through_to_pinch_grabbing(self):
        tracker = HandTracker("Right")
        self.assertEqual(
            tracker.update((0.5, 0.5), 0.2, 1.00, paint_mode=True),
            "hover",
        )
        self.assertEqual(
            tracker.update((0.5, 0.5), 0.2, 1.01, paint_mode=True),
            "hover",
        )
        self.assertFalse(tracker.grabbing)

    def test_cursor_filters_spikes_and_predicts_along_velocity(self):
        tracker = HandTracker("Right")
        for frame, x in enumerate((0.40, 0.42, 0.44, 0.46, 0.48)):
            tracker.update(
                (x, 0.5),
                0.9,
                1.00 + frame * 0.04,
                prediction_horizon_s=0.08,
            )
        self.assertGreater(tracker.velocity[0], 0.0)
        self.assertGreater(tracker.board[0], tracker.filtered_board[0])
        self.assertLessEqual(
            math.dist(tracker.board, tracker.filtered_board), 0.0251
        )

        before_spike = tracker.board
        tracker.update(
            (0.95, 0.05), 0.9, 1.21, prediction_horizon_s=0.08
        )
        self.assertLess(math.dist(tracker.board, before_spike), 0.08)

    def test_field_drawings_only_exist_in_drawing_mode_and_clear_on_resume(self):
        state = AppState()
        event = {"handId": "Right", "boardX": 0.2, "boardY": 0.3}
        state.handle_vision_event({"type": "draw_start", **event})
        self.assertEqual(state.drawings, [])

        state.set_drawing_mode(True)
        self.assertFalse(state.playing)
        state.handle_vision_event({"type": "draw_start", **event})
        state.handle_vision_event({
            "type": "draw_move", "handId": "Right",
            "boardX": 0.4, "boardY": 0.5,
        })
        state.handle_vision_event({
            "type": "draw_end", "handId": "Right",
            "boardX": 0.4, "boardY": 0.5,
        })
        self.assertEqual(len(state.drawings), 1)
        self.assertTrue(state.drawings[0]["complete"])
        self.assertEqual(len(state.drawings[0]["points"]), 2)

        state.set_drawing_mode(False)
        self.assertEqual(len(state.drawings), 1)

        state.set_playing(True)
        self.assertEqual(state.drawings, [])

    def test_drawing_mode_suppresses_player_grabs_and_is_in_snapshots(self):
        state = AppState()
        player = state.players[0]
        state.set_drawing_mode(True)
        state.handle_vision_event({
            "type": "grab_start",
            "handId": "Right",
            "boardX": player.x / 105.0,
            "boardY": player.y / 68.0,
        })

        self.assertEqual(state.grabbed, {})
        self.assertFalse(state.edit_mode)
        self.assertTrue(state.snapshot()["drawingMode"])
        self.assertTrue(state.vision_snapshot()["drawingMode"])

    def test_eraser_removes_only_the_area_it_passes_over(self):
        state = AppState()
        state.set_drawing_mode(True)
        state.drawings = [{
            "id": 1,
            "handId": "Right",
            "complete": True,
            "points": [
                [0.10, 0.50], [0.30, 0.50], [0.50, 0.50],
                [0.70, 0.50], [0.90, 0.50],
            ],
        }]
        state.handle_vision_event({
            "type": "erase_start", "handId": "Right",
            "boardX": 0.50, "boardY": 0.50,
        })

        self.assertEqual(len(state.drawings), 2)
        self.assertEqual(state.drawings[0]["points"], [[0.10, 0.50], [0.30, 0.50]])
        self.assertEqual(state.drawings[1]["points"], [[0.70, 0.50], [0.90, 0.50]])

        before = list(state.drawings)
        state.handle_vision_event({
            "type": "clear_drawings", "handId": "Right",
            "boardX": 0.50, "boardY": 0.50,
        })
        self.assertEqual(state.drawings, before)

    def test_grab_snaps_to_a_player_one_piece_width_away(self):
        state = AppState.__new__(AppState)
        state.players = [SimpleNamespace(id="H7", x=10.0, y=10.0)]
        state.grabbed = {}
        self.assertEqual(state.nearest_player(13.9, 10.0).id, "H7")
        self.assertIsNone(state.nearest_player(14.1, 10.0))


if __name__ == "__main__":
    unittest.main()
