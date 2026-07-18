from __future__ import annotations

import math

PINCH_CLOSE_RATIO = 0.38
PINCH_RELEASE_RATIO = 0.52
GESTURE_DEBOUNCE_FRAMES = 2
SMOOTH_ALPHA = 0.70
MAX_JUMP = 0.25
JUMP_REJECT_LIMIT = 2
STALE_MS = 300.0


def _pixel(landmarks, index: int, width: int, height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x * width, landmark.y * height


def pinch_pointer(
    landmarks, width: int, height: int
) -> tuple[tuple[float, float], float]:
    """Return the midpoint of both fingertips and a hand-scale pinch ratio."""

    thumb_tip = _pixel(landmarks, 4, width, height)
    index_tip = _pixel(landmarks, 8, width, height)
    wrist = _pixel(landmarks, 0, width, height)
    middle_mcp = _pixel(landmarks, 9, width, height)
    palm_size = max(1.0, math.dist(wrist, middle_mcp))
    cursor = (
        (thumb_tip[0] + index_tip[0]) / 2.0,
        (thumb_tip[1] + index_tip[1]) / 2.0,
    )
    return cursor, math.dist(thumb_tip, index_tip) / palm_size


class HandTracker:
    def __init__(self, hand_id: str) -> None:
        self.hand_id = hand_id
        self.grabbing = False
        self.pinch_ratio = float("inf")
        self.pending = 0
        self.rejects = 0
        self.missing = 0
        self.board: tuple[float, float] | None = None
        self.last_seen = 0.0

    def update(
        self, board_pt: tuple[float, float], pinch_ratio: float, now: float
    ) -> str | None:
        stale = (now - self.last_seen) * 1000.0 > STALE_MS
        self.last_seen = now

        if self.board is None or stale:
            self.board = board_pt
            self.rejects = 0
        else:
            dx = board_pt[0] - self.board[0]
            dy = board_pt[1] - self.board[1]
            if math.hypot(dx, dy) > MAX_JUMP and self.rejects < JUMP_REJECT_LIMIT:
                self.rejects += 1
                return "grab_move" if self.grabbing else "hover"
            if self.rejects >= JUMP_REJECT_LIMIT:
                self.board = board_pt
                self.rejects = 0
            else:
                self.rejects = 0
                self.board = (
                    self.board[0] + SMOOTH_ALPHA * dx,
                    self.board[1] + SMOOTH_ALPHA * dy,
                )

        self.pinch_ratio = pinch_ratio
        wants = self.grabbing
        if pinch_ratio <= PINCH_CLOSE_RATIO:
            wants = True
        elif pinch_ratio >= PINCH_RELEASE_RATIO:
            wants = False

        if wants != self.grabbing:
            self.pending += 1
            if self.pending >= GESTURE_DEBOUNCE_FRAMES:
                self.grabbing = wants
                self.pending = 0
                return "grab_start" if wants else "grab_end"
        else:
            self.pending = 0

        return "grab_move" if self.grabbing else "hover"
