from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARKER_IDS = (20, 21, 22, 23, 24, 25)
REQUIRED_MARKER_IDS = (20, 22, 23, 25)
DICTIONARY_NAME = "DICT_4X4_50"
DICTIONARY_ID = cv2.aruco.DICT_4X4_50
FIELD_ASPECT = 105.0 / 68.0
FIELD_MARKER_SIZE_X = 0.10
FIELD_MARKER_SIZE_Y = FIELD_MARKER_SIZE_X * FIELD_ASPECT
FIELD_MARKER_QUIET_X = 0.015
FIELD_MARKER_QUIET_Y = FIELD_MARKER_QUIET_X * FIELD_ASPECT
FIELD_MARKER_EDGE_X = 0.02
FIELD_MARKER_EDGE_Y = FIELD_MARKER_EDGE_X * FIELD_ASPECT


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


@dataclass(frozen=True)
class FieldMarkerPlacement:
    """An ArUco marker positioned in normalized 105 x 68 field coordinates."""

    marker_id: int
    x: float
    y: float
    width: float
    height: float
    quiet_x: float
    quiet_y: float

    @property
    def corners(self) -> np.ndarray:
        return np.asarray(
            [
                [self.x, self.y],
                [self.x + self.width, self.y],
                [self.x + self.width, self.y + self.height],
                [self.x, self.y + self.height],
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "markerId": self.marker_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "quietX": self.quiet_x,
            "quietY": self.quiet_y,
        }


def create_field_marker_layout() -> list[FieldMarkerPlacement]:
    """Return the single marker layout shared by the server and web renderers.

    X/Y values are relative to the painted touchlines. Width and height differ
    numerically so every marker remains square on a 105:68 rendered field.
    """

    xs = (
        FIELD_MARKER_EDGE_X + FIELD_MARKER_QUIET_X,
        (1.0 - FIELD_MARKER_SIZE_X) / 2.0,
        1.0 - FIELD_MARKER_EDGE_X - FIELD_MARKER_QUIET_X - FIELD_MARKER_SIZE_X,
    )
    ys = (
        FIELD_MARKER_EDGE_Y + FIELD_MARKER_QUIET_Y,
        1.0 - FIELD_MARKER_EDGE_Y - FIELD_MARKER_QUIET_Y - FIELD_MARKER_SIZE_Y,
    )
    return [
        FieldMarkerPlacement(
            marker_id=MARKER_IDS[row * 3 + column],
            x=x,
            y=y,
            width=FIELD_MARKER_SIZE_X,
            height=FIELD_MARKER_SIZE_Y,
            quiet_x=FIELD_MARKER_QUIET_X,
            quiet_y=FIELD_MARKER_QUIET_Y,
        )
        for row, y in enumerate(ys)
        for column, x in enumerate(xs)
    ]


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
