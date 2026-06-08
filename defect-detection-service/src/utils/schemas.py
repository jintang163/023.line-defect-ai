from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
import numpy as np
import time
import uuid


class DefectType(Enum):
    SCRATCH = "scratch"
    DIRT = "dirt"
    DENT = "dent"
    CRACK = "crack"
    MISSING = "missing"
    STAIN = "stain"
    DEFORMATION = "deformation"
    BUBBLE = "bubble"
    UNKNOWN = "unknown"


class DefectSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    WARNING = "warning"


class AlgorithmType(Enum):
    EDGE_DETECTION = "edge_detection"
    TEMPLATE_MATCHING = "template_matching"
    GRAY_DIFF = "gray_diff"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"


class InferenceBackend(Enum):
    ONNX_CPU = "onnx_cpu"
    ONNX_GPU = "onnx_gpu"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class DetectionResult(Enum):
    OK = "OK"
    NG = "NG"


class AlertAction(Enum):
    NONE = "none"
    LOG = "log"
    WARN = "warn"
    REJECT = "reject"
    STOP_LINE = "stop_line"


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int
    enabled: bool = True
    name: str = ""

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "enabled": self.enabled,
            "name": self.name
        }


@dataclass
class Point:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "area": self.area
        }


@dataclass
class Defect:
    defect_id: str
    type: DefectType
    severity: DefectSeverity
    confidence: float
    bbox: BoundingBox
    contour: List[Point] = field(default_factory=list)
    area_pixels: float = 0.0
    area_mm2: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, defect_type: DefectType, severity: DefectSeverity,
               confidence: float, bbox: BoundingBox,
               contour: Optional[List[Point]] = None,
               area_pixels: float = 0.0, area_mm2: float = 0.0,
               description: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> "Defect":
        return cls(
            defect_id=str(uuid.uuid4()),
            type=defect_type,
            severity=severity,
            confidence=confidence,
            bbox=bbox,
            contour=contour or [],
            area_pixels=area_pixels,
            area_mm2=area_mm2,
            description=description,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "type": self.type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "contour": [p.to_tuple() for p in self.contour],
            "area_pixels": self.area_pixels,
            "area_mm2": self.area_mm2,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class AlgorithmConfig:
    type: AlgorithmType
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    model_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "enabled": self.enabled,
            "params": self.params,
            "model_path": self.model_path
        }


@dataclass
class DefectTypeConfig:
    type: DefectType
    enabled: bool = True
    min_area_mm2: float = 0.0
    max_area_mm2: float = float('inf')
    min_confidence: float = 0.5
    severity: DefectSeverity = DefectSeverity.MAJOR
    alert_action: AlertAction = AlertAction.REJECT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "enabled": self.enabled,
            "min_area_mm2": self.min_area_mm2,
            "max_area_mm2": self.max_area_mm2,
            "min_confidence": self.min_confidence,
            "severity": self.severity.value,
            "alert_action": self.alert_action.value
        }


@dataclass
class ProductConfig:
    product_id: str
    product_name: str
    pixel_to_mm_ratio: float = 0.01
    rois: List[ROI] = field(default_factory=list)
    algorithms: List[AlgorithmConfig] = field(default_factory=list)
    defect_types: List[DefectTypeConfig] = field(default_factory=list)
    sensitivity: float = 0.8
    allowed_error_mm: float = 0.1
    allow_multiple_defects: bool = False
    max_defects_allowed: int = 0
    inference_backend: InferenceBackend = InferenceBackend.ONNX_CPU
    gpu_device_id: int = 0
    enable_tensorrt: bool = False
    inference_timeout_ms: int = 100

    def get_defect_config(self, defect_type: DefectType) -> Optional[DefectTypeConfig]:
        for dt in self.defect_types:
            if dt.type == defect_type:
                return dt
        return None

    def get_algorithm_config(self, algo_type: AlgorithmType) -> Optional[AlgorithmConfig]:
        for algo in self.algorithms:
            if algo.type == algo_type and algo.enabled:
                return algo
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "pixel_to_mm_ratio": self.pixel_to_mm_ratio,
            "rois": [r.to_dict() for r in self.rois],
            "algorithms": [a.to_dict() for a in self.algorithms],
            "defect_types": [d.to_dict() for d in self.defect_types],
            "sensitivity": self.sensitivity,
            "allowed_error_mm": self.allowed_error_mm,
            "allow_multiple_defects": self.allow_multiple_defects,
            "max_defects_allowed": self.max_defects_allowed,
            "inference_backend": self.inference_backend.value,
            "gpu_device_id": self.gpu_device_id,
            "enable_tensorrt": self.enable_tensorrt,
            "inference_timeout_ms": self.inference_timeout_ms
        }


@dataclass
class ImageData:
    image_id: str
    camera_id: str
    camera_position: str
    image: np.ndarray
    width: int = 0
    height: int = 0
    pixel_format: str = "BGR"
    timestamp: float = field(default_factory=time.time)
    sequence_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, camera_id: str, camera_position: str, image: np.ndarray,
               width: int = 0, height: int = 0, pixel_format: str = "BGR",
               sequence_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> "ImageData":
        h, w = image.shape[:2]
        return cls(
            image_id=str(uuid.uuid4()),
            camera_id=camera_id,
            camera_position=camera_position,
            image=image,
            width=width or w,
            height=height or h,
            pixel_format=pixel_format,
            timestamp=time.time(),
            sequence_id=sequence_id or str(uuid.uuid4()),
            metadata=metadata or {}
        )


@dataclass
class InferenceResult:
    success: bool
    inference_time_ms: float = 0.0
    algorithm_type: AlgorithmType = AlgorithmType.EDGE_DETECTION
    raw_output: Any = None
    error_message: str = ""


@dataclass
class DetectionOutput:
    detection_id: str
    sequence_id: str
    product_id: str
    result: DetectionResult
    defects: List[Defect] = field(default_factory=list)
    image_data: Optional[ImageData] = None
    annotated_image: Optional[np.ndarray] = None
    total_inference_time_ms: float = 0.0
    algorithm_times: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    line_id: str = ""
    station_id: str = ""
    alert_action: AlertAction = AlertAction.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, sequence_id: str, product_id: str,
               result: DetectionResult = DetectionResult.OK,
               defects: Optional[List[Defect]] = None,
               image_data: Optional[ImageData] = None,
               annotated_image: Optional[np.ndarray] = None,
               line_id: str = "", station_id: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> "DetectionOutput":
        return cls(
            detection_id=str(uuid.uuid4()),
            sequence_id=sequence_id,
            product_id=product_id,
            result=result,
            defects=defects or [],
            image_data=image_data,
            annotated_image=annotated_image,
            timestamp=time.time(),
            line_id=line_id,
            station_id=station_id,
            metadata=metadata or {}
        )

    @property
    def critical_defects(self) -> List[Defect]:
        return [d for d in self.defects if d.severity == DefectSeverity.CRITICAL]

    @property
    def major_defects(self) -> List[Defect]:
        return [d for d in self.defects if d.severity == DefectSeverity.MAJOR]

    @property
    def minor_defects(self) -> List[Defect]:
        return [d for d in self.defects if d.severity == DefectSeverity.MINOR]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "sequence_id": self.sequence_id,
            "product_id": self.product_id,
            "result": self.result.value,
            "defects": [d.to_dict() for d in self.defects],
            "total_inference_time_ms": self.total_inference_time_ms,
            "algorithm_times": self.algorithm_times,
            "timestamp": self.timestamp,
            "line_id": self.line_id,
            "station_id": self.station_id,
            "alert_action": self.alert_action.value,
            "metadata": self.metadata
        }


@dataclass
class AlertMessage:
    alert_id: str
    level: str
    category: str
    message: str
    source: str
    action: AlertAction
    timestamp: float
    detection_id: str = ""
    defect_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, level: str, category: str, message: str, source: str,
               action: AlertAction = AlertAction.LOG,
               detection_id: str = "", defect_id: str = "",
               details: Optional[Dict[str, Any]] = None) -> "AlertMessage":
        return cls(
            alert_id=str(uuid.uuid4()),
            level=level,
            category=category,
            message=message,
            source=source,
            action=action,
            timestamp=time.time(),
            detection_id=detection_id,
            defect_id=defect_id,
            details=details or {}
        )
