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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "source": self.source,
            "action": self.action.value,
            "timestamp": self.timestamp,
            "detection_id": self.detection_id,
            "defect_id": self.defect_id,
            "details": self.details
        }


class PLCProtocol(Enum):
    MODBUS_TCP = "modbus_tcp"
    OPC_UA = "opc_ua"


class PLCCommandType(Enum):
    REJECT = "reject"
    STOP_LINE = "stop_line"
    ALARM = "alarm"
    RESET = "reset"
    HEARTBEAT = "heartbeat"


@dataclass
class PLCCommand:
    command_id: str
    command_type: PLCCommandType
    timestamp: float
    detection_id: str = ""
    defect_codes: List[int] = field(default_factory=list)
    coil_address: int = 0
    value: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, command_type: PLCCommandType, detection_id: str = "",
               defect_codes: Optional[List[int]] = None,
               coil_address: int = 0, value: bool = False,
               details: Optional[Dict[str, Any]] = None) -> "PLCCommand":
        return cls(
            command_id=str(uuid.uuid4()),
            command_type=command_type,
            timestamp=time.time(),
            detection_id=detection_id,
            defect_codes=defect_codes or [],
            coil_address=coil_address,
            value=value,
            details=details or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type.value,
            "timestamp": self.timestamp,
            "detection_id": self.detection_id,
            "defect_codes": self.defect_codes,
            "coil_address": self.coil_address,
            "value": self.value,
            "details": self.details
        }


@dataclass
class PLCCommandResult:
    command_id: str
    success: bool
    timestamp: float
    response_time_ms: float = 0.0
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "success": self.success,
            "timestamp": self.timestamp,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "details": self.details
        }


@dataclass
class YieldSnapshot:
    snapshot_id: str
    product_id: str
    timestamp: float
    total_count: int
    ok_count: int
    ng_count: int
    yield_rate: float
    period_start: float
    period_end: float
    defect_distribution: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, product_id: str, total_count: int, ok_count: int, ng_count: int,
               period_start: float, period_end: float,
               defect_distribution: Optional[Dict[str, int]] = None,
               details: Optional[Dict[str, Any]] = None) -> "YieldSnapshot":
        yield_rate = (ok_count / total_count * 100) if total_count > 0 else 0.0
        return cls(
            snapshot_id=str(uuid.uuid4()),
            product_id=product_id,
            timestamp=time.time(),
            total_count=total_count,
            ok_count=ok_count,
            ng_count=ng_count,
            yield_rate=yield_rate,
            period_start=period_start,
            period_end=period_end,
            defect_distribution=defect_distribution or {},
            details=details or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "product_id": self.product_id,
            "timestamp": self.timestamp,
            "total_count": self.total_count,
            "ok_count": self.ok_count,
            "ng_count": self.ng_count,
            "yield_rate": self.yield_rate,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "defect_distribution": self.defect_distribution,
            "details": self.details
        }


@dataclass
class ProductionStats:
    product_id: str
    total_count: int = 0
    ok_count: int = 0
    ng_count: int = 0
    consecutive_ng_count: int = 0
    max_consecutive_ng: int = 0
    current_batch_start: float = 0.0
    last_snapshot_time: float = 0.0
    defect_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def yield_rate(self) -> float:
        return (self.ok_count / self.total_count * 100) if self.total_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "total_count": self.total_count,
            "ok_count": self.ok_count,
            "ng_count": self.ng_count,
            "consecutive_ng_count": self.consecutive_ng_count,
            "max_consecutive_ng": self.max_consecutive_ng,
            "yield_rate": self.yield_rate,
            "current_batch_start": self.current_batch_start,
            "last_snapshot_time": self.last_snapshot_time,
            "defect_distribution": self.defect_distribution
        }


class ManualOverrideAction(Enum):
    FORCE_PASS = "force_pass"
    FORCE_REJECT = "force_reject"
    NORMAL = "normal"


@dataclass
class ManualOverrideRecord:
    override_id: str
    detection_id: str
    action: ManualOverrideAction
    operator: str
    reason: str
    timestamp: float
    original_result: DetectionResult
    final_result: DetectionResult
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, detection_id: str, action: ManualOverrideAction,
               operator: str, reason: str,
               original_result: DetectionResult, final_result: DetectionResult,
               details: Optional[Dict[str, Any]] = None) -> "ManualOverrideRecord":
        return cls(
            override_id=str(uuid.uuid4()),
            detection_id=detection_id,
            action=action,
            operator=operator,
            reason=reason,
            timestamp=time.time(),
            original_result=original_result,
            final_result=final_result,
            details=details or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_id": self.override_id,
            "detection_id": self.detection_id,
            "action": self.action.value,
            "operator": self.operator,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "original_result": self.original_result.value,
            "final_result": self.final_result.value,
            "details": self.details
        }


class ActionLogType(Enum):
    DETECTION_RESULT = "detection_result"
    PLC_COMMAND = "plc_command"
    ALERT = "alert"
    MANUAL_OVERRIDE = "manual_override"
    SYSTEM = "system"


@dataclass
class ActionLogEntry:
    log_id: str
    log_type: ActionLogType
    timestamp: float
    level: str
    message: str
    source: str
    product_id: str = ""
    detection_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, log_type: ActionLogType, level: str, message: str, source: str,
               product_id: str = "", detection_id: str = "",
               details: Optional[Dict[str, Any]] = None) -> "ActionLogEntry":
        return cls(
            log_id=str(uuid.uuid4()),
            log_type=log_type,
            timestamp=time.time(),
            level=level,
            message=message,
            source=source,
            product_id=product_id,
            detection_id=detection_id,
            details=details or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "log_type": self.log_type.value,
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "product_id": self.product_id,
            "detection_id": self.detection_id,
            "details": self.details
        }


@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 3306
    database: str = "defect_db"
    username: str = ""
    password: str = ""
    table_prefix: str = "defect_"
    enable_timescaledb: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": "***",
            "table_prefix": self.table_prefix,
            "enable_timescaledb": self.enable_timescaledb
        }
