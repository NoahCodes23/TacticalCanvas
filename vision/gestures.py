from __future__ import annotations

import math

PINCH_CLOSE_RATIO = 0.38
PINCH_RELEASE_RATIO = 0.58
PINCH_GRAB_FRAMES = 2
PINCH_RELEASE_FRAMES = 4
INDEX_PINCH_MARGIN = 1.10
SMOOTH_ALPHA = 0.70
MAX_JUMP = 0.25
JUMP_REJECT_LIMIT = 2
STALE_MS = 300.0


def _pixel(landmarks, index: int, width: int, height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x * width, landmark.y * height


def _point3(landmarks, index: int) -> tuple[float, float, float]:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y), float(landmark.z)


def _pixel3(
    landmarks, index: int, width: int, height: int
) -> tuple[float, float, float]:
    landmark = landmarks[index]
    # MediaPipe's normalized z uses roughly the same scale as x.
    return landmark.x * width, landmark.y * height, landmark.z * width


def pinch_pointer(
    landmarks, width: int, height: int, world_landmarks=None
) -> tuple[tuple[float, float], float, bool]:
    """Return pointer, hand-scale pinch ratio, and index-pinch intent.

    The pointer always comes from image landmarks so it stays registered with
    the projected surface. Pinch intent prefers MediaPipe's metric 3D world
    landmarks, which makes the threshold much less sensitive to hand depth,
    camera angle, and lens perspective.
    """

    thumb_tip = _pixel(landmarks, 4, width, height)
    index_tip = _pixel(landmarks, 8, width, height)
    cursor = (
        (thumb_tip[0] + index_tip[0]) / 2.0,
        (thumb_tip[1] + index_tip[1]) / 2.0,
    )

    metric_points = world_landmarks if world_landmarks else landmarks
    point = _point3 if world_landmarks else (
        lambda points, index: _pixel3(points, index, width, height)
    )
    thumb = point(metric_points, 4)
    fingertips = [point(metric_points, index) for index in (8, 12, 16, 20)]
    index_mcp = point(metric_points, 5)
    pinky_mcp = point(metric_points, 17)
    wrist = point(metric_points, 0)
    middle_mcp = point(metric_points, 9)

    # Using two palm axes avoids a tiny denominator when the palm is edge-on.
    palm_size = max(
        math.dist(index_mcp, pinky_mcp),
        0.75 * math.dist(wrist, middle_mcp),
        1e-6,
    )
    tip_distances = [math.dist(thumb, fingertip) for fingertip in fingertips]
    index_distance = tip_distances[0]
    nearest_other = min(tip_distances[1:])
    index_is_primary = index_distance <= nearest_other * INDEX_PINCH_MARGIN
    return cursor, index_distance / palm_size, index_is_primary


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
        self,
        board_pt: tuple[float, float],
        pinch_ratio: float,
        now: float,
        index_is_primary: bool = True,
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
        # Accidental thumb contact with another fingertip must not start a grab.
        # Once grabbed, release is based only on opening the intended pinch so
        # crossing another finger through the cursor cannot drop the piece.
        if pinch_ratio <= PINCH_CLOSE_RATIO and (
            self.grabbing or index_is_primary
        ):
            wants = True
        elif pinch_ratio >= PINCH_RELEASE_RATIO:
            wants = False

        if wants != self.grabbing:
            self.pending += 1
            required_frames = (
                PINCH_GRAB_FRAMES if wants else PINCH_RELEASE_FRAMES
            )
            if self.pending >= required_frames:
                self.grabbing = wants
                self.pending = 0
                return "grab_start" if wants else "grab_end"
        else:
            self.pending = 0

        return "grab_move" if self.grabbing else "hover"
