from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import numpy as np
import time
import uuid


class CameraType(Enum):
    BASLER = "basler"
    HIKVISION = "hikvision"
    MOCK = "mock"


class CameraStatus(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    INITIALIZING = "initializing"
    ERROR = "error"
    CAPTURING = "capturing"


class LightMode(Enum):
    STROBE = "strobe"
    CONTINUOUS = "continuous"
    OFF = "off"


class TriggerMode(Enum):
    EXTERNAL = "external"
    SOFTWARE = "software"
    CONTINUOUS = "continuous"


@dataclass
class CameraConfig:
    id: str
    name: str
    type: CameraType
    position: str
    ip: str
    serial_number: str
    exposure_time: int
    gain: float
    width: int
    height: int
    pixel_format: str
    trigger_mode: TriggerMode
    enabled: bool = True


@dataclass
class LightChannelConfig:
    id: str
    name: str
    mode: LightMode
    brightness: int
    color_temp: int
    strobe_delay: int = 0
    strobe_width: int = 1000


@dataclass
class CameraStatusInfo:
    camera_id: str
    status: CameraStatus
    exposure_time: int
    gain: float
    trigger_count: int
    last_capture_time: Optional[float] = None
    temperature: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CapturedImage:
    image_id: str
    camera_id: str
    camera_position: str
    timestamp: float
    raw_data: np.ndarray
    processed_data: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    trigger_count: int = 0
    metadata: Dict = field(default_factory=dict)
    sequence_id: str = ""

    @classmethod
    def create(cls, camera_id: str, camera_position: str, raw_data: np.ndarray,
               width: int, height: int, pixel_format: str, trigger_count: int = 0,
               sequence_id: str = "", metadata: Optional[Dict] = None) -> "CapturedImage":
        return cls(
            image_id=str(uuid.uuid4()),
            camera_id=camera_id,
            camera_position=camera_position,
            timestamp=time.time(),
            raw_data=raw_data,
            width=width,
            height=height,
            pixel_format=pixel_format,
            trigger_count=trigger_count,
            sequence_id=sequence_id or str(uuid.uuid4()),
            metadata=metadata or {}
        )


@dataclass
class ImageMessage:
    sequence_id: str
    timestamp: float
    images: List[CapturedImage]
    product_id: Optional[str] = None
    line_id: str = "line-001"

    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "product_id": self.product_id,
            "line_id": self.line_id,
            "images": [
                {
                    "image_id": img.image_id,
                    "camera_id": img.camera_id,
                    "camera_position": img.camera_position,
                    "timestamp": img.timestamp,
                    "width": img.width,
                    "height": img.height,
                    "pixel_format": img.pixel_format,
                    "trigger_count": img.trigger_count,
                    "metadata": img.metadata
                }
                for img in self.images
            ]
        }


@dataclass
class AlertMessage:
    alert_id: str
    level: str
    category: str
    message: str
    source: str
    timestamp: float
    details: Dict = field(default_factory=dict)

    @classmethod
    def create(cls, level: str, category: str, message: str, source: str,
               details: Optional[Dict] = None) -> "AlertMessage":
        return cls(
            alert_id=str(uuid.uuid4()),
            level=level,
            category=category,
            message=message,
            source=source,
            timestamp=time.time(),
            details=details or {}
        )


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DistortionParams:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    @classmethod
    def from_config(cls, config: Dict) -> "DistortionParams":
        return cls(
            camera_matrix=np.array(config["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(config["dist_coeffs"], dtype=np.float64)
        )
