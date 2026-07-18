from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Size:
    width: int
    height: int


@dataclass(frozen=True)
class DisplayInfo:
    index: int
    x: int
    y: int
    width: int
    height: int
    name: str = "Display"
    is_primary: bool = False


@dataclass
class ProjectorCalibration:
    """Everything an application needs to work in projector coordinates."""

    camera_size: Size
    projector_size: Size
    display: DisplayInfo
    camera_index: int
    camera_to_projector: list[list[float]]
    projector_to_camera: list[list[float]]
    reprojection_rmse: float
    camera_jitter: float
    markers_used: list[int]
    camera_fps: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)
    dictionary: str = "DICT_4X4_50"
    version: int = 1
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def map_points(
        self,
        points: Sequence[Point] | np.ndarray,
        direction: Literal["camera_to_projector", "projector_to_camera"] = "camera_to_projector",
    ) -> np.ndarray:
        if isinstance(points, np.ndarray):
            source = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        else:
            source = np.asarray([[point.x, point.y] for point in points], dtype=np.float64)
        matrix = self.camera_to_projector if direction == "camera_to_projector" else self.projector_to_camera
        return cv2.perspectiveTransform(source.reshape(-1, 1, 2), np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)

    def camera_point_to_projector(self, x: float, y: float) -> Point:
        mapped = self.map_points(np.asarray([[x, y]]), "camera_to_projector")[0]
        return Point(float(mapped[0]), float(mapped[1]))

    def projector_point_to_camera(self, x: float, y: float) -> Point:
        mapped = self.map_points(np.asarray([[x, y]]), "projector_to_camera")[0]
        return Point(float(mapped[0]), float(mapped[1]))

    def projector_corners_in_camera(self) -> np.ndarray:
        corners = np.asarray(
            [
                [0, 0],
                [self.projector_size.width - 1, 0],
                [self.projector_size.width - 1, self.projector_size.height - 1],
                [0, self.projector_size.height - 1],
            ],
            dtype=np.float64,
        )
        return self.map_points(corners, "projector_to_camera")

    def add_info(self, key: str, value: Any) -> None:
        """Attach application-specific serializable information."""
        self.metadata[key] = value

    def get_info(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProjectorCalibration:
        data = dict(value)
        data["camera_size"] = Size(**data["camera_size"])
        data["projector_size"] = Size(**data["projector_size"])
        data["display"] = DisplayInfo(**data["display"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> ProjectorCalibration:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Backwards-friendly name for code written against the first prototype.
CalibrationResult = ProjectorCalibration
