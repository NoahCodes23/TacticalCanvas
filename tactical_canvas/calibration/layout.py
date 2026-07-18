from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARKER_IDS = (20, 21, 22, 23, 24, 25)
REQUIRED_MARKER_IDS = (20, 22, 23, 25)
DICTIONARY_NAME = "DICT_4X4_50"
DICTIONARY_ID = cv2.aruco.DICT_4X4_50


@dataclass(frozen=True)
class MarkerPlacement:
    marker_id: int
    x: int
    y: int
    size: int
    quiet_zone: int

    @property
    def corners(self) -> np.ndarray:
        return np.asarray(
            [
                [self.x, self.y],
                [self.x + self.size, self.y],
                [self.x + self.size, self.y + self.size],
                [self.x, self.y + self.size],
            ],
            dtype=np.float32,
        )


def create_marker_layout(width: int, height: int, bottom_reserved: int = 0) -> list[MarkerPlacement]:
    usable_height = height - bottom_reserved
    if width < 480 or usable_height < 320:
        raise ValueError("The calibration display must be at least 480 x 320 pixels")

    marker_size = round(min(220, width * 0.13, usable_height * 0.20))
    quiet_zone = max(12, round(marker_size * 0.16))
    edge = max(20, round(min(width, usable_height) * 0.035))
    xs = (
        edge + quiet_zone,
        round((width - marker_size) / 2),
        width - edge - quiet_zone - marker_size,
    )
    ys = (
        edge + quiet_zone,
        usable_height - edge - quiet_zone - marker_size,
    )

    placements: list[MarkerPlacement] = []
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            placements.append(
                MarkerPlacement(
                    marker_id=MARKER_IDS[row * 3 + column],
                    x=x,
                    y=y,
                    size=marker_size,
                    quiet_zone=quiet_zone,
                )
            )
    return placements


def render_calibration_pattern(width: int, height: int, bottom_reserved: int = 0) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    for placement in create_marker_layout(width, height, bottom_reserved=bottom_reserved):
        marker = cv2.aruco.generateImageMarker(dictionary, placement.marker_id, placement.size)
        image[
            placement.y : placement.y + placement.size,
            placement.x : placement.x + placement.size,
        ] = marker
    return image
