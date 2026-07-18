"""Shared webcam opening rules for the TacticalCanvas rig."""

from __future__ import annotations

import os
import sys

import cv2


def configured_camera_index(camera_index: int | None = None) -> int:
    if camera_index is not None:
        return camera_index
    try:
        return int(os.environ.get("TC_CAMERA", "1"))
    except ValueError as error:
        raise ValueError("TC_CAMERA must be an integer camera index") from error


def open_webcam(camera_index: int | None = None) -> tuple[cv2.VideoCapture, int]:
    """Open the configured webcam without forcing any capture properties."""
    resolved_index = configured_camera_index(camera_index)
    capture = (
        cv2.VideoCapture(resolved_index, cv2.CAP_DSHOW)
        if sys.platform == "win32"
        else cv2.VideoCapture(resolved_index)
    )
    return capture, resolved_index
