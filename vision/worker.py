from __future__ import annotations

import json 
import math
import multiprocessing
import os
import queue as queue_mod
import threading
import time
from dataclasses import dataclass

import cv2

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from tactical_canvas.calibration.detector import ArucoMarkerDetector
from tactical_canvas.calibration.layout import (
    MARKER_IDS,
    create_field_marker_layout,
)
from tactical_canvas.calibration.models import (
    FieldCalibration,
    ProjectorCalibration,
    Size,
    field_warp_basis,
)
from tactical_canvas.calibration.solver import CalibrationAccumulator
from vision.gestures import HandTracker, pinch_pointer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "hand_landmarker.task")
CALIB_PATH = os.path.join(ROOT, "cache", "calibration.json")
PROJECTOR_CALIB_PATH = os.path.join(ROOT, "cache", "projector-calibration.json")
FIELD_CALIB_PATH = os.path.join(ROOT, "cache", "field-calibration.json")

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = float(os.environ.get("TC_CAMERA_FPS", "60"))

DETECT_WIDTH = int(os.environ.get("TC_DETECT_WIDTH", "480")) or None
LANDMARK_INPUT_PX = 224
HAND_LOST_FRAMES = 6
GRABBING_HAND_LOST_FRAMES = 15
CALIBRATION_TIMEOUT_S = 30.0
CALIBRATION_MAX_JITTER_PX = 6.0
CALIBRATION_MAX_FIELD_RMSE = 0.02

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
        self.camera_size: tuple[int, int] | None = None
        self.field_correction: np.ndarray | None = None
        self.source = "none"

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
        self.camera_size = None
        self.field_correction = None
        self.source = "manual"
        print("[vision] calibrated. press 's' to save.")

    def reset(self) -> None:
        self.points = []
        self.H = None
        self.collecting = True
        self.camera_size = None
        self.field_correction = None
        self.source = "none"
        print(f"[vision] click the {CORNER_NAMES[0]} pitch corner")

    def to_board(
        self,
        px: float,
        py: float,
        frame_size: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        if self.camera_size is not None and frame_size is not None:
            calibrated_width, calibrated_height = self.camera_size
            frame_width, frame_height = frame_size
            if frame_width > 1 and frame_height > 1:
                px *= (calibrated_width - 1) / (frame_width - 1)
                py *= (calibrated_height - 1) / (frame_height - 1)
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        bx, by = float(out[0, 0, 0]), float(out[0, 0, 1])
        if self.field_correction is not None and self.field_correction.shape == (2, 10):
            basis = field_warp_basis(np.asarray([[bx, by]]))[0]
            bx, by = np.asarray([bx, by]) + self.field_correction @ basis
        return float(bx), float(by)

    def save(self) -> None:
        if self.H is None:
            print("[vision] nothing to save -- calibrate first (press 'c)")
            return
        
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump({"H": self.H.tolist(), "points": self.points}, f, indent=2)
        print(f"[vision] saved {CALIB_PATH}")

    def load(self) -> bool:
        if os.path.exists(FIELD_CALIB_PATH):
            try:
                calibration = FieldCalibration.load(FIELD_CALIB_PATH)
                self.H = np.asarray(calibration.camera_to_field, dtype=np.float64)
                self.camera_size = (
                    calibration.camera_size.width,
                    calibration.camera_size.height,
                )
                coefficients = np.asarray(
                    calibration.correction_coefficients, dtype=np.float64
                )
                self.field_correction = (
                    coefficients if coefficients.shape == (2, 10) else None
                )
                corners = np.asarray(
                    [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64
                )
                mapped = cv2.perspectiveTransform(
                    corners.reshape(-1, 1, 2),
                    np.asarray(calibration.field_to_camera, dtype=np.float64),
                ).reshape(-1, 2)
                self.points = [
                    tuple(int(round(value)) for value in point) for point in mapped
                ]
                self.collecting = False
                self.source = "aruco-field"
                print(f"[vision] loaded field calibration from {FIELD_CALIB_PATH}")
                return True
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"[vision] could not load field calibration: {error}")

        if os.path.exists(PROJECTOR_CALIB_PATH):
            try:
                calibration = ProjectorCalibration.load(PROJECTOR_CALIB_PATH)
                projector_width = max(1, calibration.projector_size.width - 1)
                projector_height = max(1, calibration.projector_size.height - 1)
                normalize_projector = np.array(
                    [
                        [1.0 / projector_width, 0.0, 0.0],
                        [0.0, 1.0 / projector_height, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                self.H = normalize_projector @ np.asarray(
                    calibration.camera_to_projector,
                    dtype=np.float64,
                )
                self.camera_size = (
                    calibration.camera_size.width,
                    calibration.camera_size.height,
                )
                self.field_correction = None
                corners = np.rint(calibration.projector_corners_in_camera()).astype(int)
                self.points = [tuple(int(value) for value in point) for point in corners]
                self.collecting = False
                self.source = "aruco"
                print(f"[vision] loaded ArUco calibration from {PROJECTOR_CALIB_PATH}")
                return True
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"[vision] could not load ArUco calibration: {error}")

        if not os.path.exists(CALIB_PATH):
            return False
        try:
            with open(CALIB_PATH) as f:
                data = json.load(f)
            self.H = np.array(data["H"], dtype=np.float32)
            self.points = [tuple(p) for p in data.get("points", [])]
            self.camera_size = None
            self.field_correction = None
            self.source = "manual"
            print(f"[vision] loaded calibration from {CALIB_PATH}")
            return True
        except Exception as e:
            print(f"[vision] count not load calibration: {e}")
            return False
        
def hand_size_px(landmarks, w: int, h: int) -> float:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DEFAULT_CAMERA_FPS)

    return cap


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    image: np.ndarray
    captured_at_ms: float
    captured_perf: float


class LatestFrameCapture:
    """Continuously capture and expose only the newest available camera frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._sequence = 0
        self._stopping = False
        self.failures = 0
        self.frames = 0
        self._fps = 0.0
        self._last_frame_perf = 0.0
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="tc-latest-camera",
            daemon=True,
        )

    def start(self) -> LatestFrameCapture:
        self._thread.start()
        return self

    @property
    def fps(self) -> float:
        return self._fps

    def _capture_loop(self) -> None:
        while not self._stopping:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.failures += 1
                if self.failures > 120:
                    break
                time.sleep(0.002)
                continue
            self.failures = 0
            self.frames += 1
            self._sequence += 1
            captured_perf = time.perf_counter()
            if self._last_frame_perf:
                frame_dt = captured_perf - self._last_frame_perf
                if frame_dt > 0:
                    instant_fps = 1.0 / frame_dt
                    self._fps = (
                        0.9 * self._fps + 0.1 * instant_fps
                        if self._fps
                        else instant_fps
                    )
            self._last_frame_perf = captured_perf
            packet = CapturedFrame(
                sequence=self._sequence,
                image=frame,
                captured_at_ms=time.time() * 1000.0,
                captured_perf=captured_perf,
            )
            with self._condition:
                self._latest = packet
                self._condition.notify_all()

        with self._condition:
            self._condition.notify_all()

    def read_after(
        self, sequence: int, timeout: float = 0.25
    ) -> CapturedFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._stopping
                or (self._latest is not None and self._latest.sequence > sequence)
                or not self._thread.is_alive(),
                timeout=timeout,
            )
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            return self._latest

    def close(self) -> None:
        self._stopping = True
        with self._condition:
            self._condition.notify_all()
        self.capture.release()
        self._thread.join(timeout=2)

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

    calibration_status = calib.source.upper() if calib.ready else "NO"
    status = (f"FPS {fps:4.1f} | hands {hand_count} | "
              f"calib {calibration_status} | "
              f"hand {hand_px:3.0f}px/{LANDMARK_INPUT_PX}")
    cv2.putText(frame, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for t in trackers.values():
        if t.board is None:
            continue
        colour = (0, 255, 0) if t.grabbing else (200, 200, 200)
        off = "" if 0 <= t.board[0] <= 1 and 0 <= t.board[1] <= 1 else " off-pitch"
        label = (f"{t.hand_id} pinch {t.pinch_ratio:.2f} {'GRAB' if t.grabbing else 'open'} "
                 f"({t.board[0]:.2f}, {t.board[1]:.2f}){off}")
        cv2.putText(frame, label, (10, 56 + 24 * list(trackers).index(t.hand_id)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

def run(
    event_queue=None,
    camera_index: int = 1,
    show_preview: bool = False,
    control_queue=None,
) -> None:
    try:
        _run_loop(event_queue, camera_index, show_preview, control_queue)
    except KeyboardInterrupt:
        print("[vision] interrupted")


def _run_loop(event_queue, camera_index: int, show_preview: bool, control_queue) -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing hand landmarker model: {MODEL_PATH}")

    calib = Calibration()
    calib.load()

    cap = open_camera(camera_index)
    camera_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(
        f"[vision] camera {camera_index}: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {camera_fps:.0f}fps requested {DEFAULT_CAMERA_FPS:.0f}fps"
    )

    def emit(evt: dict) -> None:
        if event_queue is None:
            if evt.get("type") != "vision_stats":
                print("[vision]", evt)
            return
        try:
            event_queue.put_nowait(evt)
        except queue_mod.Full:
            # Never let old cursor events create an input backlog.
            try:
                event_queue.get_nowait()
                event_queue.put_nowait(evt)
            except (queue_mod.Empty, queue_mod.Full):
                pass
        except Exception:
            pass

    parent = multiprocessing.parent_process()
    trackers: dict[str, HandTracker] = {}
    state_lock = threading.Lock()
    pending_lock = threading.Lock()
    pending: dict[int, dict] = {}

    hand_px = 0.0
    inference_fps = 0.0
    last_result_perf = 0.0
    last_stats_perf = 0.0
    last_hand_count = 0
    last_landmarks = []
    inference_ms = 0.0
    capture_to_result_ms = 0.0
    capture_drops = 0
    submitted_frames = 0
    completed_frames = 0
    overlay = show_preview
    calibration_active = False
    calibration_started = 0.0
    calibration_last_status = 0.0
    calibration_detector: ArucoMarkerDetector | None = None
    calibration_accumulator: CalibrationAccumulator | None = None

    def emit_calibration_status(phase: str, **details) -> None:
        emit({
            "type": "calibration_status",
            "phase": phase,
            "active": phase in ("starting", "scanning", "solving"),
            **details,
        })

    def process_calibration_commands() -> None:
        nonlocal calibration_active, calibration_started
        nonlocal calibration_detector, calibration_accumulator
        if control_queue is None:
            return
        while True:
            try:
                command = control_queue.get_nowait()
            except (queue_mod.Empty, InterruptedError):
                return
            except Exception:
                return
            command_type = command.get("type") if isinstance(command, dict) else None
            if command_type == "start_calibration":
                calibration_detector = ArucoMarkerDetector()
                calibration_accumulator = CalibrationAccumulator(
                    create_field_marker_layout(),
                    minimum_samples=14,
                    required_marker_ids=MARKER_IDS,
                )
                calibration_started = time.monotonic()
                calibration_active = True
                with state_lock:
                    trackers.clear()
                emit_calibration_status(
                    "starting", progress=0.0, visibleMarkers=0, requiredMarkers=6
                )
            elif command_type == "cancel_calibration":
                calibration_active = False
                calibration_detector = None
                calibration_accumulator = None
                emit_calibration_status("cancelled", progress=0.0)

    def on_result(result, _output_image, timestamp_ms: int) -> None:
        nonlocal hand_px, inference_fps, last_result_perf, last_stats_perf
        nonlocal last_hand_count, last_landmarks, inference_ms
        nonlocal capture_to_result_ms, completed_frames

        completed_perf = time.perf_counter()
        completed_at_ms = time.time() * 1000.0
        with pending_lock:
            metadata = pending.pop(timestamp_ms, None)
        if metadata is None:
            return

        completed_frames += 1
        current_inference_ms = (completed_perf - metadata["submitted_perf"]) * 1000.0
        current_pipeline_ms = (completed_perf - metadata["captured_perf"]) * 1000.0
        inference_ms = (
            0.8 * inference_ms + 0.2 * current_inference_ms
            if inference_ms
            else current_inference_ms
        )
        capture_to_result_ms = (
            0.8 * capture_to_result_ms + 0.2 * current_pipeline_ms
            if capture_to_result_ms
            else current_pipeline_ms
        )
        if last_result_perf:
            result_dt = completed_perf - last_result_perf
            if result_dt > 0:
                instant_fps = 1.0 / result_dt
                inference_fps = (
                    0.85 * inference_fps + 0.15 * instant_fps
                    if inference_fps
                    else instant_fps
                )
        last_result_perf = completed_perf

        w = metadata["width"]
        h = metadata["height"]
        detect_w = metadata["detect_width"]
        detect_h = metadata["detect_height"]
        captured_at_ms = metadata["captured_at_ms"]
        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0
        seen: set[str] = set()

        with state_lock:
            last_hand_count = hand_count
            last_landmarks = result.hand_landmarks or []
            hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0
            if result.hand_landmarks:
                biggest = max(hand_size_px(l, detect_w, detect_h)
                              for l in result.hand_landmarks)
                hand_px = 0.8 * hand_px + 0.2 * biggest if hand_px else biggest

            if result.hand_landmarks and calib.ready and not calibration_active:
                for landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness, strict=False
                ):
                    hand_id = handedness[0].category_name
                    if hand_id in seen:
                        continue     
                    seen.add(hand_id)

                    pointer, pinch_ratio = pinch_pointer(landmarks, w, h)
                    bx, by = calib.to_board(*pointer, frame_size=(w, h))

                    if not (math.isfinite(bx) and math.isfinite(by)):
                        continue
                    if abs(bx) > BOARD_LIMIT or abs(by) > BOARD_LIMIT:
                        continue

                    tracker = trackers.setdefault(hand_id, HandTracker(hand_id))
                    tracker.missing = 0
                    etype = tracker.update((bx, by), pinch_ratio, completed_perf)
                    if etype and tracker.board:
                        emit({
                            "type": etype,
                            "handId": hand_id,
                            "boardX": tracker.board[0],
                            "boardY": tracker.board[1],
                            "confidence": float(handedness[0].score),
                            "capturedAtMs": captured_at_ms,
                            "inferenceCompletedAtMs": completed_at_ms,
                            "inferenceMs": round(current_inference_ms, 2),
                            "captureToInferenceMs": round(current_pipeline_ms, 2),
                        })

            for hand_id, tracker in list(trackers.items()):
                if hand_id in seen:
                    continue
                tracker.missing += 1
                missing_limit = (
                    GRABBING_HAND_LOST_FRAMES
                    if tracker.grabbing
                    else HAND_LOST_FRAMES
                )
                if tracker.missing >= missing_limit:
                    emit({
                        "type": "hand_lost",
                        "handId": hand_id,
                        "capturedAtMs": captured_at_ms,
                    })
                    trackers.pop(hand_id, None)

        if completed_perf - last_stats_perf > 0.5:
            last_stats_perf = completed_perf
            emit({
                "type": "vision_stats",
                "fps": round(inference_fps, 1),
                "captureFps": round(camera_stream.fps, 1),
                "hands": hand_count,
                "calibrated": calib.ready,
                "handPx": round(hand_px),
                "inferenceMs": round(inference_ms, 1),
                "captureToInferenceMs": round(capture_to_result_ms, 1),
                "captureDrops": capture_drops,
                "submittedFrames": submitted_frames,
                "completedFrames": completed_frames,
                "cameraFps": round(camera_fps, 1),
                "capturedAtMs": captured_at_ms,
            })

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    window = "TacticalCanvas vision"
    if show_preview:
        cv2.namedWindow(window)
        cv2.setMouseCallback(
            window,
            lambda e, x, y, f, p: (
                calib.add_point(x, y) if e == cv2.EVENT_LBUTTONDOWN else None
            ),
        )

    camera_stream = LatestFrameCapture(cap).start()
    last_sequence = 0
    last_timestamp_ms = 0
    loop_count = 0

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                loop_count += 1
                if (
                    parent is not None
                    and loop_count % 30 == 0
                    and not parent.is_alive()
                ):
                    print("[vision] server is gone; releasing the camera and exiting")
                    break

                packet = camera_stream.read_after(last_sequence)
                if packet is None:
                    if not camera_stream._thread.is_alive():
                        print("[vision] camera stopped delivering frames; exiting")
                        break
                    continue

                capture_drops += max(0, packet.sequence - last_sequence - 1)
                last_sequence = packet.sequence
                frame = packet.image
                h, w = frame.shape[:2]
                process_calibration_commands()
                if (
                    calibration_active
                    and calibration_detector is not None
                    and calibration_accumulator is not None
                ):
                    detected = calibration_detector.detect(frame)
                    calibration_accumulator.add_frame(detected)
                    required_visible = len(set(detected).intersection(MARKER_IDS))
                    visible_markers = len(set(detected).intersection(MARKER_IDS))
                    status_now = time.monotonic()
                    if status_now - calibration_last_status >= 0.1:
                        calibration_last_status = status_now
                        emit_calibration_status(
                            "scanning",
                            progress=round(calibration_accumulator.progress, 3),
                            visibleMarkers=visible_markers,
                            requiredVisible=required_visible,
                            requiredMarkers=len(MARKER_IDS),
                        )
                    if calibration_accumulator.ready:
                        emit_calibration_status(
                            "solving", progress=1.0, visibleMarkers=visible_markers
                        )
                        try:
                            field_calibration = calibration_accumulator.solve_field(
                                camera_size=Size(width=w, height=h),
                                camera_index=camera_index,
                                camera_fps=camera_fps,
                            )
                            if (
                                field_calibration.camera_jitter
                                > CALIBRATION_MAX_JITTER_PX
                                or field_calibration.reprojection_rmse
                                > CALIBRATION_MAX_FIELD_RMSE
                            ):
                                raise RuntimeError(
                                    "Calibration was unstable; keep the camera still and retry"
                                )
                            field_calibration.save(FIELD_CALIB_PATH)
                            with state_lock:
                                calib.H = np.asarray(
                                    field_calibration.camera_to_field,
                                    dtype=np.float64,
                                )
                                calib.camera_size = (w, h)
                                coefficients = np.asarray(
                                    field_calibration.correction_coefficients,
                                    dtype=np.float64,
                                )
                                calib.field_correction = (
                                    coefficients
                                    if coefficients.shape == (2, 10)
                                    else None
                                )
                                calib.points = []
                                calib.collecting = False
                                calib.source = "aruco-field"
                                trackers.clear()
                            calibration_active = False
                            calibration_detector = None
                            calibration_accumulator = None
                            emit_calibration_status(
                                "succeeded",
                                progress=1.0,
                                calibrated=True,
                                markersUsed=field_calibration.markers_used,
                                reprojectionRmse=round(
                                    field_calibration.reprojection_rmse, 5
                                ),
                                cameraJitter=round(
                                    field_calibration.camera_jitter, 2
                                ),
                                lensCorrection=bool(
                                    field_calibration.correction_coefficients
                                ),
                            )
                        except (OSError, RuntimeError, ValueError) as error:
                            calibration_active = False
                            calibration_detector = None
                            calibration_accumulator = None
                            emit_calibration_status(
                                "failed", progress=0.0, reason=str(error)
                            )
                    elif status_now - calibration_started >= CALIBRATION_TIMEOUT_S:
                        final_progress = calibration_accumulator.progress
                        calibration_active = False
                        calibration_detector = None
                        calibration_accumulator = None
                        emit_calibration_status(
                            "failed",
                            progress=round(final_progress, 3),
                            reason="Timed out before all six field markers were stable",
                        )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if DETECT_WIDTH and w > DETECT_WIDTH:
                    scale = DETECT_WIDTH / w
                    small = cv2.resize(
                        rgb,
                        (DETECT_WIDTH, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = rgb

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                timestamp_ms = max(
                    last_timestamp_ms + 1, int(packet.captured_perf * 1000.0)
                )
                last_timestamp_ms = timestamp_ms
                metadata = {
                    "captured_at_ms": packet.captured_at_ms,
                    "captured_perf": packet.captured_perf,
                    "submitted_perf": time.perf_counter(),
                    "width": w,
                    "height": h,
                    "detect_width": small.shape[1],
                    "detect_height": small.shape[0],
                    # Keep the backing array alive until MediaPipe completes.
                    "mp_image": mp_image,
                }
                with pending_lock:
                    pending[timestamp_ms] = metadata
                    if len(pending) > 64:
                        for old_timestamp in sorted(pending)[:-32]:
                            pending.pop(old_timestamp, None)
                submitted_frames += 1
                landmarker.detect_async(mp_image, timestamp_ms)

                if show_preview:
                    preview = frame.copy()
                    with state_lock:
                        preview_landmarks = list(last_landmarks)
                        preview_trackers = dict(trackers)
                        preview_fps = inference_fps
                        preview_hand_count = last_hand_count
                        preview_hand_px = hand_px
                    if overlay:
                        for landmarks in preview_landmarks:
                            pts = [
                                (int(lm.x * w), int(lm.y * h))
                                for lm in landmarks
                            ]
                            for a, b in HAND_CONNECTIONS:
                                cv2.line(
                                    preview, pts[a], pts[b], (255, 255, 255), 2
                                )
                            for point in pts:
                                cv2.circle(preview, point, 3, (0, 0, 255), -1)
                        draw_overlay(
                            preview,
                            calib,
                            preview_trackers,
                            preview_fps,
                            preview_hand_count,
                            preview_hand_px,
                        )

                    cv2.imshow(window, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c"):
                        calib.reset()
                    elif key == ord("r"):
                        calib.points = []
                        calib.collecting = True
                    elif key == ord("s"):
                        calib.save()
                    elif key == ord("h"):
                        overlay = not overlay
    finally:
        camera_stream.close()
        if show_preview:
            cv2.destroyAllWindows()
    print("[vision] stopped")


if __name__ == "__main__":
    run(None, camera_index=int(os.environ.get("TC_CAMERA", "1")))
