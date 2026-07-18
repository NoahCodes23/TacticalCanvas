from __future__ import annotations

import math
from collections import deque

PINCH_CLOSE_RATIO = 0.45
PINCH_RELEASE_RATIO = 0.72
WORLD_PINCH_CLOSE_RATIO = 0.38
WORLD_PINCH_RELEASE_RATIO = 0.58
PINCH_GRAB_FRAMES = 2
PINCH_RELEASE_FRAMES = 4
DRAW_START_FRAMES = 3
DRAW_RELEASE_FRAMES = 3
DRAW_TIP_GAP_RATIO = 0.28
DRAW_OTHER_FINGER_MARGIN = 0.72
DRAW_MIN_JOINT_ANGLE = 145.0
DRAW_MIN_PALM_FACING = 0.48
POSITION_ALPHA_IDLE = 0.18
POSITION_ALPHA_MOVING = 0.68
VELOCITY_ALPHA = 0.32
POSITION_DEADZONE = 0.0025
FULL_SPEED_BOARD_S = 0.75
MAX_FILTER_RESIDUAL = 0.10
MAX_PREDICTION_HORIZON_S = 0.10
MAX_PREDICTION_LEAD = 0.025
STALE_MS = 300.0


def _pixel(landmarks, index: int, width: int, height: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x * width, landmark.y * height


def _point3(landmarks, index: int) -> tuple[float, float, float]:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y), float(landmark.z)


def _vector(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(end[i] - start[i] for i in range(3))


def _joint_angle(a, b, c) -> float:
    ba = _vector(b, a)
    bc = _vector(b, c)
    denominator = math.sqrt(sum(v * v for v in ba)) * math.sqrt(
        sum(v * v for v in bc)
    )
    if denominator <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(ba, bc)) / denominator))
    return math.degrees(math.acos(cosine))


def drawing_pointer(
    landmarks, width: int, height: int, world_landmarks=None
) -> tuple[tuple[float, float], bool]:
    """Return the two-finger pen tip and whether its deliberate pose is active."""

    index_tip_px = _pixel(landmarks, 8, width, height)
    middle_tip_px = _pixel(landmarks, 12, width, height)
    pointer = (
        (index_tip_px[0] + middle_tip_px[0]) / 2.0,
        (index_tip_px[1] + middle_tip_px[1]) / 2.0,
    )

    points = world_landmarks or landmarks
    wrist = _point3(points, 0)
    index_mcp = _point3(points, 5)
    pinky_mcp = _point3(points, 17)
    palm_size = max(math.dist(index_mcp, pinky_mcp), 1e-6)
    index_tip = _point3(points, 8)
    middle_tip = _point3(points, 12)
    ring_tip = _point3(points, 16)

    tip_gap = math.dist(index_tip, middle_tip) / palm_size
    middle_ring_gap = math.dist(middle_tip, ring_tip) / palm_size
    separated_from_ring = tip_gap <= middle_ring_gap * DRAW_OTHER_FINGER_MARGIN

    index_straight = (
        _joint_angle(_point3(points, 5), _point3(points, 6), _point3(points, 8))
        >= DRAW_MIN_JOINT_ANGLE
        and _joint_angle(_point3(points, 6), _point3(points, 7), index_tip)
        >= DRAW_MIN_JOINT_ANGLE
    )
    middle_straight = (
        _joint_angle(_point3(points, 9), _point3(points, 10), middle_tip)
        >= DRAW_MIN_JOINT_ANGLE
        and _joint_angle(_point3(points, 10), _point3(points, 11), middle_tip)
        >= DRAW_MIN_JOINT_ANGLE
    )

    palm_a = _vector(wrist, index_mcp)
    palm_b = _vector(wrist, pinky_mcp)
    normal = (
        palm_a[1] * palm_b[2] - palm_a[2] * palm_b[1],
        palm_a[2] * palm_b[0] - palm_a[0] * palm_b[2],
        palm_a[0] * palm_b[1] - palm_a[1] * palm_b[0],
    )
    normal_length = math.sqrt(sum(value * value for value in normal))
    palm_facing = abs(normal[2]) / max(normal_length, 1e-9)

    active = (
        tip_gap <= DRAW_TIP_GAP_RATIO
        and separated_from_ring
        and index_straight
        and middle_straight
        and palm_facing >= DRAW_MIN_PALM_FACING
    )
    return pointer, active


def pinch_pointer(
    landmarks, width: int, height: int, world_landmarks=None
) -> tuple[tuple[float, float], float, float | None]:
    """Return pointer plus visible and 3D thumb-index pinch ratios.

    The visible image ratio is authoritative: what looks closed to the user is
    a pinch, and what looks open is a release. The world ratio only resolves
    the hysteresis band between those two states.
    """

    thumb_tip = _pixel(landmarks, 4, width, height)
    index_tip = _pixel(landmarks, 8, width, height)
    cursor = (
        (thumb_tip[0] + index_tip[0]) / 2.0,
        (thumb_tip[1] + index_tip[1]) / 2.0,
    )

    image_palm_size = max(
        math.dist(_pixel(landmarks, 5, width, height),
                  _pixel(landmarks, 17, width, height)),
        0.75 * math.dist(_pixel(landmarks, 0, width, height),
                         _pixel(landmarks, 9, width, height)),
        1.0,
    )
    image_ratio = math.dist(thumb_tip, index_tip) / image_palm_size

    world_ratio = None
    if world_landmarks:
        world_palm_size = max(
            math.dist(_point3(world_landmarks, 5),
                      _point3(world_landmarks, 17)),
            0.75 * math.dist(_point3(world_landmarks, 0),
                             _point3(world_landmarks, 9)),
            1e-6,
        )
        world_ratio = (
            math.dist(_point3(world_landmarks, 4),
                      _point3(world_landmarks, 8))
            / world_palm_size
        )

    return cursor, image_ratio, world_ratio


class HandTracker:
    def __init__(self, hand_id: str) -> None:
        self.hand_id = hand_id
        self.grabbing = False
        self.drawing = False
        self.pinch_ratio = float("inf")
        self.pending = 0
        self.draw_pending = 0
        self.draw_release_pending = 0
        self.missing = 0
        self.board: tuple[float, float] | None = None
        self.filtered_board: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.position_history: deque[tuple[float, float]] = deque(maxlen=3)
        self.last_seen = 0.0

    def _update_position(
        self,
        board_pt: tuple[float, float],
        now: float,
        prediction_horizon_s: float,
    ) -> None:
        stale = (now - self.last_seen) * 1000.0 > STALE_MS
        dt = max(1.0 / 120.0, min(0.12, now - self.last_seen))
        self.last_seen = now

        if self.filtered_board is None or stale:
            self.position_history.clear()
            self.position_history.append(board_pt)
            self.filtered_board = board_pt
            self.velocity = (0.0, 0.0)
            self.board = board_pt
            return

        self.position_history.append(board_pt)
        if len(self.position_history) < 3:
            count = len(self.position_history)
            measurement = (
                sum(point[0] for point in self.position_history) / count,
                sum(point[1] for point in self.position_history) / count,
            )
        else:
            xs = sorted(point[0] for point in self.position_history)
            ys = sorted(point[1] for point in self.position_history)
            measurement = (xs[1], ys[1])

        dx = measurement[0] - self.filtered_board[0]
        dy = measurement[1] - self.filtered_board[1]
        residual = math.hypot(dx, dy)
        if residual > MAX_FILTER_RESIDUAL:
            scale = MAX_FILTER_RESIDUAL / residual
            dx *= scale
            dy *= scale
            residual = MAX_FILTER_RESIDUAL

        speed = residual / dt
        motion = min(1.0, speed / FULL_SPEED_BOARD_S)
        alpha = POSITION_ALPHA_IDLE + (
            POSITION_ALPHA_MOVING - POSITION_ALPHA_IDLE
        ) * motion
        if residual < POSITION_DEADZONE:
            alpha *= residual / POSITION_DEADZONE

        previous = self.filtered_board
        self.filtered_board = (
            previous[0] + alpha * dx,
            previous[1] + alpha * dy,
        )
        instant_velocity = (
            (self.filtered_board[0] - previous[0]) / dt,
            (self.filtered_board[1] - previous[1]) / dt,
        )
        velocity_alpha = VELOCITY_ALPHA * (0.35 + 0.65 * motion)
        self.velocity = (
            self.velocity[0]
            + velocity_alpha * (instant_velocity[0] - self.velocity[0]),
            self.velocity[1]
            + velocity_alpha * (instant_velocity[1] - self.velocity[1]),
        )
        if residual < POSITION_DEADZONE:
            self.velocity = (self.velocity[0] * 0.55, self.velocity[1] * 0.55)

        horizon = max(0.0, min(MAX_PREDICTION_HORIZON_S, prediction_horizon_s))
        lead_x = self.velocity[0] * horizon
        lead_y = self.velocity[1] * horizon
        lead = math.hypot(lead_x, lead_y)
        if lead > MAX_PREDICTION_LEAD:
            scale = MAX_PREDICTION_LEAD / lead
            lead_x *= scale
            lead_y *= scale
        self.board = (
            self.filtered_board[0] + lead_x,
            self.filtered_board[1] + lead_y,
        )

    def update(
        self,
        board_pt: tuple[float, float],
        pinch_ratio: float,
        now: float,
        world_pinch_ratio: float | None = None,
        draw_pose: bool = False,
        prediction_horizon_s: float = 0.0,
    ) -> str | None:
        self._update_position(board_pt, now, prediction_horizon_s)

        if draw_pose:
            self.draw_release_pending = 0
            if not self.drawing:
                self.draw_pending += 1
                if self.draw_pending >= DRAW_START_FRAMES:
                    self.drawing = True
                    self.grabbing = False
                    self.pending = 0
                    self.draw_pending = 0
                    return "draw_start"
                return "hover"
            return "draw_move"

        self.draw_pending = 0
        if self.drawing:
            self.draw_release_pending += 1
            if self.draw_release_pending >= DRAW_RELEASE_FRAMES:
                self.drawing = False
                self.draw_release_pending = 0
                return "draw_end"
            return "draw_move"
        self.draw_release_pending = 0

        self.pinch_ratio = pinch_ratio
        wants = self.grabbing
        # The visible gesture wins at both extremes. Depth is useful only in
        # the ambiguous middle, where it cannot veto visibly touching tips.
        if pinch_ratio <= PINCH_CLOSE_RATIO:
            wants = True
        elif pinch_ratio >= PINCH_RELEASE_RATIO:
            wants = False
        elif world_pinch_ratio is not None:
            if world_pinch_ratio <= WORLD_PINCH_CLOSE_RATIO:
                wants = True
            elif world_pinch_ratio >= WORLD_PINCH_RELEASE_RATIO:
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
