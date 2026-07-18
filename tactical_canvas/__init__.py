"""TacticalCanvas projector calibration backend."""

from .calibration.models import FieldCalibration, ProjectorCalibration
from .webcam_calibration import CalibrationError, calibrate_webcam

__version__ = "0.1.0"

__all__ = [
    "CalibrationError",
    "FieldCalibration",
    "ProjectorCalibration",
    "calibrate_webcam",
]
