from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from screeninfo import get_monitors

from .camera import open_webcam
from .calibration.detector import ArucoMarkerDetector
from .calibration.layout import MARKER_IDS, REQUIRED_MARKER_IDS, create_marker_layout, render_calibration_pattern
from .calibration.models import DisplayInfo, ProjectorCalibration, Size
from .calibration.solver import CalibrationAccumulator

WINDOW_NAME = "TacticalCanvas | Webcam calibration"
DEFAULT_SAVE_PATH = Path("data/projector-calibration.json")
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30.0


class CalibrationError(RuntimeError):
    """Raised when automatic projector calibration cannot complete."""


class CalibrationCancelled(CalibrationError):
    """Raised when the user presses Escape or closes the marker window."""


def available_displays() -> list[DisplayInfo]:
    """Return displays in the same index order accepted by calibrate_webcam()."""
    return [
        DisplayInfo(
            index=index,
            x=int(monitor.x),
            y=int(monitor.y),
            width=int(monitor.width),
            height=int(monitor.height),
            name=str(monitor.name or f"Display {index}"),
            is_primary=bool(monitor.is_primary),
        )
        for index, monitor in enumerate(get_monitors())
    ]


def select_projector_display(monitor_index: int | None = None) -> DisplayInfo:
    displays = available_displays()
    if not displays:
        raise CalibrationError("No displays were detected")
    if monitor_index is not None:
        if monitor_index < 0 or monitor_index >= len(displays):
            raise CalibrationError(f"Display index {monitor_index} does not exist")
        return displays[monitor_index]
    return next((display for display in displays if not display.is_primary), displays[0])


def request_camera_mode(
    capture: cv2.VideoCapture,
    width: int | None,
    height: int | None,
    fps: float | None,
) -> None:
    """Request a capture mode while allowing callers to opt into negotiation."""
    if (width is None) != (height is None):
        raise ValueError("camera width and height must both be set or both be None")
    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        capture.set(cv2.CAP_PROP_FPS, fps)


def render_calibration_window(
    width: int,
    height: int,
    message: str,
    *,
    message_kind: str = "error",
    status_bar_height: int | None = None,
    base_frame: np.ndarray | None = None,
    webcam_preview: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Build the exact fullscreen image shown by OpenCV."""
    bar_height = status_bar_height or min(
        max(150, round(height * 0.18)),
        max(80, height - 320),
    )
    if base_frame is None:
        grayscale = render_calibration_pattern(width, height, bottom_reserved=bar_height)
        frame = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    else:
        frame = base_frame.copy()
    frame[height - bar_height :, :] = (18, 21, 27)
    colors = {
        "error": (90, 100, 255),
        "working": (220, 220, 220),
        "success": (176, 242, 117),
    }
    color = colors.get(message_kind, colors["working"])
    font_scale = max(0.55, min(0.85, width / 1900))
    cv2.putText(
        frame,
        message,
        (24, height - max(17, round(bar_height * 0.32))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    if webcam_preview is not None:
        available_height = max(60, bar_height - 18)
        preview_width = min(340, max(120, width // 4))
        preview_height = round(preview_width * webcam_preview.shape[0] / webcam_preview.shape[1])
        if preview_height > available_height:
            preview_height = available_height
            preview_width = round(preview_height * webcam_preview.shape[1] / webcam_preview.shape[0])
        preview = cv2.resize(webcam_preview, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        x = width - preview_width - 10
        y = height - preview_height - 9
        cv2.rectangle(frame, (x - 3, y - 3), (x + preview_width + 3, y + preview_height + 3), (235, 235, 235), 2)
        frame[y : y + preview_height, x : x + preview_width] = preview
    return frame, bar_height


def render_debug_webcam(
    camera_frame: np.ndarray,
    detected: dict[int, np.ndarray],
    camera_index: int,
) -> np.ndarray:
    """Annotate the raw camera POV with exactly what the detector accepted."""
    preview = camera_frame.copy()
    for marker_id, corners in detected.items():
        polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [polygon], True, (80, 255, 140), 3, cv2.LINE_AA)
        anchor = tuple(int(value) for value in polygon[0, 0])
        cv2.putText(
            preview,
            f"ID {marker_id}",
            (anchor[0], max(22, anchor[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 255, 140),
            2,
            cv2.LINE_AA,
        )
    ids = ",".join(str(marker_id) for marker_id in sorted(detected)) or "none"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 34), (18, 21, 27), -1)
    cv2.putText(
        preview,
        f"WEBCAM {camera_index} | {preview.shape[1]}x{preview.shape[0]} | DETECTED IDS: {ids}",
        (9, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return preview


def calibrate_webcam(
    camera_index: int | None = None,
    monitor_index: int | None = None,
    *,
    camera_width: int | None = DEFAULT_CAMERA_WIDTH,
    camera_height: int | None = DEFAULT_CAMERA_HEIGHT,
    camera_fps: float | None = DEFAULT_CAMERA_FPS,
    minimum_samples: int = 18,
    timeout_seconds: float = 60,
    warmup_seconds: float = 1.0,
    save_path: str | Path | None = DEFAULT_SAVE_PATH,
    metadata: dict[str, Any] | None = None,
) -> ProjectorCalibration:
    """
    Display ArUco markers on the projector and calibrate an invisible webcam feed.

    The first non-primary display is selected automatically. Pass monitor_index to
    override it. Escape cancels. On success, a ProjectorCalibration object is
    returned and optionally saved as JSON.
    """
    try:
        capture, camera_index = open_webcam(camera_index)
        request_camera_mode(capture, camera_width, camera_height, camera_fps)
    except ValueError as error:
        raise CalibrationError(str(error)) from error
    display = select_projector_display(monitor_index)
    projector_size = Size(display.width, display.height)
    status_frame, status_bar_height = render_calibration_window(
        display.width,
        display.height,
        "STARTING: Opening webcam in the background...",
        message_kind="working",
    )
    base_frame = status_frame.copy()
    layout = create_marker_layout(display.width, display.height, bottom_reserved=status_bar_height)
    accumulator = CalibrationAccumulator(layout, minimum_samples=minimum_samples)
    window_created = False
    last_debug_preview: np.ndarray | None = None

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.moveWindow(WINDOW_NAME, display.x, display.y)
        cv2.resizeWindow(WINDOW_NAME, display.width, display.height)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW_NAME, status_frame)
        cv2.waitKey(1)

        if not capture.isOpened():
            raise CalibrationError(f"Webcam {camera_index} could not be opened")
        detector = ArucoMarkerDetector()
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        effective_camera_fps = reported_fps if reported_fps > 0 else (camera_fps or DEFAULT_CAMERA_FPS)
        started_at = time.monotonic()
        camera_size: Size | None = None

        while True:
            elapsed = time.monotonic() - started_at
            if elapsed > timeout_seconds:
                raise CalibrationError("Timed out: the webcam could not see all four projector corners")
            ok, camera_frame = capture.read()
            if not ok or camera_frame is None:
                raise CalibrationError("The webcam stopped returning frames")
            camera_size = Size(width=int(camera_frame.shape[1]), height=int(camera_frame.shape[0]))
            detected = detector.detect(camera_frame)
            visible_markers = len(set(detected).intersection(MARKER_IDS))
            visible_corners = len(set(detected).intersection(REQUIRED_MARKER_IDS))
            last_debug_preview = render_debug_webcam(camera_frame, detected, camera_index)
            if elapsed >= warmup_seconds:
                accumulator.add_frame(detected)

            if visible_corners < len(REQUIRED_MARKER_IDS):
                message = (
                    f"ERROR: {visible_markers}/6 markers visible | {visible_corners}/4 required corners | "
                    "aim camera at full screen | ESC cancels"
                )
                kind = "error"
            else:
                message = (
                    f"CALIBRATING: {visible_markers}/6 markers visible | 4/4 required corners | "
                    f"hold still {round(accumulator.progress * 100)}% | ESC cancels"
                )
                kind = "working"
            output, _ = render_calibration_window(
                display.width,
                display.height,
                message,
                message_kind=kind,
                status_bar_height=status_bar_height,
                base_frame=base_frame,
                webcam_preview=last_debug_preview,
            )
            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                raise CalibrationCancelled("Calibration cancelled")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                raise CalibrationCancelled("Calibration window was closed")

            if accumulator.ready:
                calibration = accumulator.solve(
                    camera_size=camera_size,
                    projector_size=projector_size,
                    display=display,
                    camera_index=camera_index,
                    camera_fps=effective_camera_fps,
                )
                if calibration.camera_jitter > 6 or calibration.reprojection_rmse > 8:
                    raise CalibrationError("Calibration was unstable; keep the webcam still and retry")
                if metadata:
                    calibration.metadata.update(metadata)
                if save_path is not None:
                    calibration.save(save_path)
                success, _ = render_calibration_window(
                    display.width,
                    display.height,
                    (
                        f"SUCCESS: Projector mapped | fit {calibration.reprojection_rmse:.2f}px | "
                        f"camera jitter {calibration.camera_jitter:.2f}px"
                    ),
                    message_kind="success",
                    status_bar_height=status_bar_height,
                    base_frame=base_frame,
                    webcam_preview=last_debug_preview,
                )
                cv2.imshow(WINDOW_NAME, success)
                cv2.waitKey(700)
                return calibration
    except Exception as error:
        if window_created:
            failure, _ = render_calibration_window(
                display.width,
                display.height,
                f"ERROR: {error}",
                message_kind="error",
                status_bar_height=status_bar_height,
                base_frame=base_frame,
                webcam_preview=last_debug_preview,
            )
            cv2.imshow(WINDOW_NAME, failure)
            cv2.waitKey(1800)
        if isinstance(error, CalibrationError):
            raise
        raise CalibrationError(str(error)) from error
    finally:
        if capture is not None:
            capture.release()
        if window_created:
            cv2.destroyWindow(WINDOW_NAME)
