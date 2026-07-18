from __future__ import annotations

import cv2
import numpy as np

from .layout import DICTIONARY_ID


class ArucoMarkerDetector:
    """Thin, injectable wrapper around OpenCV's ArUco detector."""

    def __init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        corners, ids, _rejected = self._detector.detectMarkers(frame)
        if ids is None:
            return {}
        # A projected webcam PiP can contain tiny recursive copies of the same
        # markers. Keep the largest quadrilateral for each ID: it is the actual
        # calibration marker, not its copy inside the debug preview.
        detected: dict[int, np.ndarray] = {}
        areas: dict[int, float] = {}
        for marker_id, marker_corners in zip(ids.flatten(), corners, strict=True):
            marker_id = int(marker_id)
            value = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            area = abs(float(cv2.contourArea(value)))
            if area > areas.get(marker_id, -1):
                detected[marker_id] = value
                areas[marker_id] = area
        return detected
