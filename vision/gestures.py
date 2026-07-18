from __future__ import annotations

import math
from collections import deque

PINCH_CLOSE_RATIO = 0.45
PINCH_RELEASE_RATIO = 0.72
WORLD_PINCH_CLOSE_RATIO = 0.38
WORLD_PINCH_RELEASE_RATIO = 0.58
PINCH_GRAB_FRAMES = 2
PINCH_RELEASE_FRAMES = 4
DRAW_START_FRAMES = 2
DRAW_RELEASE_FRAMES = 3
ERASE_START_FRAMES = 2
ERASE_RELEASE_FRAMES = 3
CLEAR_HOLD_FRAMES = 12
DRAW_TIP_GAP_RATIO = 0.36
ERASE_TIP_GAP_RATIO = 0.42
DRAW_MIN_EXTENSION_RATIO = 1.35
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


def drawing_pointer(
    landmarks, width: int, height: int, world_landmarks=None
) -> tuple[tuple[float, float], bool]:
    """Return the visible two-finger pen tip and whether it is active.

    Drawing intentionally ignores palm direction and inferred world depth. If
    the index and middle fingers visibly form one extended pen in the camera,
    it should draw whether the palm or the back of the hand faces the lens.
    """

    draw_pointer, _, draw_pose, _, _ = paint_gestures(
        landmarks, width, height
    )
    return draw_pointer, draw_pose


def _finger_extended(
    landmarks, mcp: int, pip: int, tip: int, width: int, height: int
) -> bool:
    mcp_point = _pixel(landmarks, mcp, width, height)
    pip_point = _pixel(landmarks, pip, width, height)
    tip_point = _pixel(landmarks, tip, width, height)
    proximal_length = math.dist(mcp_point, pip_point)
    return proximal_length >= 2.0 and math.dist(mcp_point, tip_point) >= (
        DRAW_MIN_EXTENSION_RATIO * proximal_length
    )


def paint_gestures(
    landmarks, width: int, height: int
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    bool,
    bool,
    bool,
]:
    """Return pen/eraser pointers and visible 2/3/5-finger tool poses."""

    index_tip = _pixel(landmarks, 8, width, height)
    middle_tip = _pixel(landmarks, 12, width, height)
    ring_tip = _pixel(landmarks, 16, width, height)
    draw_pointer = (
        (index_tip[0] + middle_tip[0]) / 2.0,
        (index_tip[1] + middle_tip[1]) / 2.0,
    )
    erase_pointer = (
        (index_tip[0] + middle_tip[0] + ring_tip[0]) / 3.0,
        (index_tip[1] + middle_tip[1] + ring_tip[1]) / 3.0,
    )

    wrist = _pixel(landmarks, 0, width, height)
    index_mcp = _pixel(landmarks, 5, width, height)
    middle_mcp = _pixel(landmarks, 9, width, height)
    pinky_mcp = _pixel(landmarks, 17, width, height)
    palm_size = max(
        math.dist(index_mcp, pinky_mcp),
        0.75 * math.dist(wrist, middle_mcp),
        1.0,
    )

    index_extended = _finger_extended(landmarks, 5, 6, 8, width, height)
    middle_extended = _finger_extended(landmarks, 9, 10, 12, width, height)
    ring_extended = _finger_extended(landmarks, 13, 14, 16, width, height)
    pinky_extended = _finger_extended(landmarks, 17, 18, 20, width, height)
    thumb_tip = _pixel(landmarks, 4, width, height)
    thumb_ip = _pixel(landmarks, 3, width, height)
    thumb_extended = (
        math.dist(thumb_tip, index_mcp) / palm_size >= 0.42
        and math.dist(thumb_tip, thumb_ip) / palm_size >= 0.15
    )
    index_middle_close = (
        math.dist(index_tip, middle_tip) / palm_size <= DRAW_TIP_GAP_RATIO
    )
    middle_ring_close = (
        math.dist(middle_tip, ring_tip) / palm_size <= ERASE_TIP_GAP_RATIO
    )

    clear_pose = all((
        thumb_extended,
        index_extended,
        middle_extended,
        ring_extended,
        pinky_extended,
    ))
    erase_pose = (
        not clear_pose
        and index_extended
        and middle_extended
        and ring_extended
        and index_middle_close
        and middle_ring_close
    )
    draw_pose = (
        not erase_pose
        and not clear_pose
        and index_extended
        and middle_extended
        and index_middle_close
    )
    return draw_pointer, erase_pointer, draw_pose, erase_pose, clear_pose


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
        self.erasing = False
        self.pinch_ratio = float("inf")
        self.pending = 0
        self.draw_pending = 0
        self.draw_release_pending = 0
        self.erase_pending = 0
        self.erase_release_pending = 0
        self.clear_pending = 0
        self.clear_latched = False
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
        erase_pose: bool = False,
        clear_pose: bool = False,
        prediction_horizon_s: float = 0.0,
    ) -> str | None:
        self._update_position(board_pt, now, prediction_horizon_s)

        if clear_pose:
            if self.drawing:
                self.drawing = False
                self.draw_pending = 0
                return "draw_end"
            if self.erasing:
                self.erasing = False
                self.erase_pending = 0
                return "erase_end"
            self.draw_pending = 0
            self.erase_pending = 0
            if not self.clear_latched:
                self.clear_pending += 1
                if self.clear_pending >= CLEAR_HOLD_FRAMES:
                    self.clear_pending = 0
                    self.clear_latched = True
                    self.grabbing = False
                    self.pending = 0
                    return "clear_drawings"
            return "hover"

        self.clear_pending = 0
        self.clear_latched = False

        if erase_pose:
            if self.drawing:
                self.drawing = False
                self.draw_pending = 0
                return "draw_end"
            self.erase_release_pending = 0
            self.draw_pending = 0
            if not self.erasing:
                self.erase_pending += 1
                if self.erase_pending >= ERASE_START_FRAMES:
                    self.erasing = True
                    self.grabbing = False
                    self.pending = 0
                    self.erase_pending = 0
                    return "erase_start"
                return "hover"
            return "erase_move"

        self.erase_pending = 0
        if self.erasing:
            self.erase_release_pending += 1
            if self.erase_release_pending >= ERASE_RELEASE_FRAMES:
                self.erasing = False
                self.erase_release_pending = 0
                return "erase_end"
            return "erase_move"
        self.erase_release_pending = 0

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
