import json 
import math
import multiprocessing
import os
import time

import cv2

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "hand_landmarker.task")
CALIB_PATH = os.path.join(ROOT, "cache", "calibration.json")

DETECT_WIDTH = None
EXTEND_RATIO = 1.15
LANDMARK_INPUT_PX = 224
GRAB_FINGERS = 1
RELEASE_FINGERS = 2
GESTURE_DEBOUNCE_FRAMES = 3
SMOOTH_ALPHA = 0.45

MAX_JUMP = 0.25
JUMP_REJECT_LIMIT = 3
STALE_MS = 300.0
HAND_LOST_FRAMES = 10

BOARD_LIMIT = 3.0
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
BOARD_CORNERS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

class Calibration:
    def __init__(self) -> None:
        self.H: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []
        self.collecting = False

    @property
    def ready(self) -> bool:
        return self.H is not None
    
    def add_point(self, x: int, y: int) -> None:
        if not self.collecting or len(self.points) >= 4:
            return
        
        self.points.append((x, y))
        print(f"[vision] corner {len(self.points)}/4 at ({x}, {y})")
        if len(self.points) == 4:
            self._solve()

    def _solve(self) -> None:
        src = np.array(self.points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, BOARD_CORNERS)
        self.collecting = False
        print("[vision] calibrated. press 's' to save.")

    def reset(self) -> None:
        self.points = []
        self.H = None
        self.collecting = True
        print(f"[vision] click the {CORNER_NAMES[0]} pitch coarner")

    def to_board(self, px: float, py: float) -> tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def save(self) -> None:
        if self.H is None:
            print("[vision] nothing to save -- calibrate first (press 'c)")
            return
        
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump({"H": self.H.tolist(), "points": self.points}, f, indent=2)
        print(f"[vision] saved {CALIB_PATH}")

    def load(self) -> bool:
        if not os.path.exists(CALIB_PATH):
            return False
        try:
            with open(CALIB_PATH) as f:
                data = json.load(f)
            self.H = np.array(data["H"], dtype=np.float32)
            self.points = [tuple(p) for p in data.get("points", [])]
            print(f"[vision] loaded calibration from {CALIB_PATH}")
            return True
        except Exception as e:
            print(f"[vision] count not load calibration: {e}")
            return False
        
class HandTracker:
    def __init__(self, hand_id: str) -> None:
        self.hand_id = hand_id
        self.grabbing = False
        self.fingers = 0
        self.pending = 0
        self.rejects = 0
        self.missing = 0
        self.board: tuple[float, float] | None = None
        self.last_seen = 0.0

    def update(self, board_pt: tuple[float, float], fingers:int, now:float) -> str | None:
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
                return "grab_move" if self.grabbing else "hover"  # hold position
            if self.rejects >= JUMP_REJECT_LIMIT:
                self.board = board_pt  # sustained: the hand really is over there
                self.rejects = 0
            else:
                self.rejects = 0
                self.board = (
                    self.board[0] + SMOOTH_ALPHA * dx,
                    self.board[1] + SMOOTH_ALPHA * dy,
                )

        self.fingers = fingers

        wants = self.grabbing
        if fingers == GRAB_FINGERS:
            wants = True
        elif fingers >= RELEASE_FINGERS:
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

def _px(landmarks, idx: int, w: int, h: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

FINGER_JOINTS = [(6, 8), (10, 12), (14, 16), (18, 20)]


def hand_size_px(landmarks, w: int, h: int) -> float:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def count_extended(world_landmarks) -> int:
    wrist = world_landmarks[0]
    n = 0
    for pip, tip in FINGER_JOINTS:
        reach = _dist3(wrist, world_landmarks[tip])
        knuckle = _dist3(wrist, world_landmarks[pip])
        if knuckle > 1e-6 and reach > knuckle * EXTEND_RATIO:
            n += 1
    return n

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")

    return cap

def draw_overlay(frame, calib, trackers, fps, hand_count, hand_px):
    h, w = frame.shape[:2]

    if calib.points:
        pts = np.array(calib.points, dtype=np.int32)
        if len(calib.points) == 4:
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        for i, (x, y) in enumerate(calib.points):
            cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if calib.collecting:
        nxt = CORNER_NAMES[len(calib.points)] if len(calib.points) < 4 else ""
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"CLICK THE {nxt} PITCH CORNER", (12, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    elif not calib.ready:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "NOT CALIBRATED - press 'c'", (12, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    status = (f"FPS {fps:4.1f} | hands {hand_count} | "
              f"calib {'OK' if calib.ready else 'NO'} | "
              f"hand {hand_px:3.0f}px/{LANDMARK_INPUT_PX}")
    cv2.putText(frame, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for t in trackers.values():
        if t.board is None:
            continue
        colour = (0, 255, 0) if t.grabbing else (200, 200, 200)
        off = "" if 0 <= t.board[0] <= 1 and 0 <= t.board[1] <= 1 else " off-pitch"
        label = (f"{t.hand_id} {t.fingers}f {'GRAB' if t.grabbing else 'open'} "
                 f"({t.board[0]:.2f}, {t.board[1]:.2f}){off}")
        cv2.putText(frame, label, (10, 56 + 24 * list(trackers).index(t.hand_id)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

def run(event_queue=None, camera_index: int = 1, show_preview: bool = True) -> None:
    try:
        _run_loop(event_queue, camera_index, show_preview)
    except KeyboardInterrupt:
        print("[vision] interrupted")

def _run_loop(event_queue, camera_index: int, show_preview: bool) -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing hand landmarker model: {MODEL_PATH}")

    calib = Calibration()
    calib.load()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = open_camera(camera_index)
    print(
        f"[vision] camera {camera_index}: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps"
    )

    window = "TacticalCanvas vision"
    if show_preview:
        cv2.namedWindow(window)
        cv2.setMouseCallback(
            window, lambda e, x, y, f, p: calib.add_point(x, y) if e == cv2.EVENT_LBUTTONDOWN else None
        )

    def emit(evt: dict) -> None:
        if event_queue is None:
            if evt.get("type") != "vision_stats":
                print("[vision]", evt)
            return
        try:
            event_queue.put_nowait(evt)
        except Exception:
            pass 

    parent = multiprocessing.parent_process()

    trackers: dict[str, HandTracker] = {}
    hand_px = 0.0
    fps = 0.0
    last_t = time.monotonic()
    last_stats = 0.0
    failures = 0
    overlay = show_preview

    frame_no = 0
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame_no += 1
            if parent is not None and frame_no % 15 == 0 and not parent.is_alive():
                print("[vision] server is gone; releasing the camera and exiting")
                break

            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 60:
                    print("[vision] camera stopped delivering frames; exiting")
                    break
                continue
            failures = 0

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if DETECT_WIDTH and w > DETECT_WIDTH:
                scale = DETECT_WIDTH / w
                small = cv2.resize(rgb, (DETECT_WIDTH, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = rgb

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
            result = landmarker.detect_for_video(mp_image, int(now * 1000))

            hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0
            seen: set[str] = set()

            detect_h, detect_w = small.shape[:2]
            if result.hand_landmarks:
                biggest = max(hand_size_px(l, detect_w, detect_h)
                              for l in result.hand_landmarks)
                hand_px = 0.8 * hand_px + 0.2 * biggest if hand_px else biggest

            if result.hand_landmarks and calib.ready:
                for landmarks, world, handedness in zip(
                    result.hand_landmarks, result.hand_world_landmarks, result.handedness
                ):
                    hand_id = handedness[0].category_name
                    if hand_id in seen:
                        continue     
                    seen.add(hand_id)

                    tip = _px(landmarks, 8, w, h) 
                    bx, by = calib.to_board(*tip)

                    if not (math.isfinite(bx) and math.isfinite(by)):
                        continue
                    if abs(bx) > BOARD_LIMIT or abs(by) > BOARD_LIMIT:
                        continue

                    fingers = count_extended(world)
                    tracker = trackers.setdefault(hand_id, HandTracker(hand_id))
                    tracker.missing = 0
                    etype = tracker.update((bx, by), fingers, now)
                    if etype and tracker.board:
                        emit({
                            "type": etype,
                            "handId": hand_id,
                            "boardX": tracker.board[0],
                            "boardY": tracker.board[1],
                            "confidence": float(handedness[0].score),
                        })

            for hand_id, tracker in list(trackers.items()):
                if hand_id in seen:
                    continue
                tracker.missing += 1
                if tracker.missing >= HAND_LOST_FRAMES:
                    emit({"type": "hand_lost", "handId": hand_id})
                    trackers.pop(hand_id, None)

            if now - last_stats > 0.5:
                last_stats = now
                emit({"type": "vision_stats", "fps": round(fps, 1),
                      "hands": hand_count, "calibrated": calib.ready,
                      "handPx": round(hand_px)})

            if show_preview:
                if overlay:
                    for landmarks in (result.hand_landmarks or []):
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
                        for p in pts:
                            cv2.circle(frame, p, 3, (0, 0, 255), -1)
                    draw_overlay(frame, calib, trackers, fps, hand_count, hand_px)

                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    calib.reset()
                elif key == ord("r"):
                    calib.points = []
                    calib.collecting = True
                elif key == ord("s"):
                    calib.save()
                elif key == ord("h"):
                    overlay = not overlay

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    print("[vision] stopped")


if __name__ == "__main__":
    run(None, camera_index=int(os.environ.get("TC_CAMERA", "1")))