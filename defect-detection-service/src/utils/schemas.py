from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
import numpy as np
import time
import uuid
from datetime import datetime


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
    sequence_id: str
    detection_id: str
    action: ManualOverrideAction
    operator: str
    reason: str
    timestamp: float
    original_result: DetectionResult
    final_result: DetectionResult
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, sequence_id: str, detection_id: str, action: ManualOverrideAction,
               operator: str, reason: str,
               original_result: DetectionResult, final_result: DetectionResult,
               details: Optional[Dict[str, Any]] = None) -> "ManualOverrideRecord":
        return cls(
            override_id=str(uuid.uuid4()),
            sequence_id=sequence_id,
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
            "sequence_id": self.sequence_id,
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


@dataclass
class DetectionRecord:
    record_id: str
    detection_id: str
    sequence_id: str
    product_id: str
    product_name: str
    product_batch: str
    product_model: str
    result: DetectionResult
    defect_types: str
    defect_count: int
    inference_time_ms: float
    model_version: str
    timestamp: float
    line_id: str = ""
    station_id: str = ""
    camera_id: str = ""
    original_image_path: str = ""
    annotated_image_path: str = ""
    thumbnail_path: str = ""
    defects_detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, detection_id: str, sequence_id: str,
               product_id: str, product_name: str,
               product_batch: str, product_model: str,
               result: DetectionResult,
               defect_types: str, defect_count: int,
               inference_time_ms: float, model_version: str,
               line_id: str = "", station_id: str = "",
               camera_id: str = "",
               original_image_path: str = "",
               annotated_image_path: str = "",
               thumbnail_path: str = "",
               defects_detail: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> "DetectionRecord":
        return cls(
            record_id=str(uuid.uuid4()),
            detection_id=detection_id,
            sequence_id=sequence_id,
            product_id=product_id,
            product_name=product_name,
            product_batch=product_batch,
            product_model=product_model,
            result=result,
            defect_types=defect_types,
            defect_count=defect_count,
            inference_time_ms=inference_time_ms,
            model_version=model_version,
            timestamp=time.time(),
            line_id=line_id,
            station_id=station_id,
            camera_id=camera_id,
            original_image_path=original_image_path,
            annotated_image_path=annotated_image_path,
            thumbnail_path=thumbnail_path,
            defects_detail=defects_detail,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "detection_id": self.detection_id,
            "sequence_id": self.sequence_id,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "product_batch": self.product_batch,
            "product_model": self.product_model,
            "result": self.result.value,
            "defect_types": self.defect_types,
            "defect_count": self.defect_count,
            "inference_time_ms": self.inference_time_ms,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat() if self.timestamp else "",
            "date_partition": datetime.fromtimestamp(self.timestamp).strftime("%Y%m%d") if self.timestamp else "",
            "line_id": self.line_id,
            "station_id": self.station_id,
            "camera_id": self.camera_id,
            "original_image_path": self.original_image_path,
            "annotated_image_path": self.annotated_image_path,
            "thumbnail_path": self.thumbnail_path,
            "defects_detail": self.defects_detail,
            "metadata": self.metadata
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "DetectionRecord":
        result_val = row.get("result", "OK")
        try:
            result = DetectionResult(result_val)
        except ValueError:
            result = DetectionResult.OK
        return cls(
            record_id=row.get("record_id", ""),
            detection_id=row.get("detection_id", ""),
            sequence_id=row.get("sequence_id", ""),
            product_id=row.get("product_id", ""),
            product_name=row.get("product_name", ""),
            product_batch=row.get("product_batch", ""),
            product_model=row.get("product_model", ""),
            result=result,
            defect_types=row.get("defect_types", ""),
            defect_count=row.get("defect_count", 0),
            inference_time_ms=row.get("inference_time_ms", 0.0),
            model_version=row.get("model_version", ""),
            timestamp=row.get("timestamp", 0.0),
            line_id=row.get("line_id", ""),
            station_id=row.get("station_id", ""),
            camera_id=row.get("camera_id", ""),
            original_image_path=row.get("original_image_path", ""),
            annotated_image_path=row.get("annotated_image_path", ""),
            thumbnail_path=row.get("thumbnail_path", ""),
            defects_detail=row.get("defects_detail", ""),
            metadata=row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        )


class ModelVersionStatus(Enum):
    DRAFT = "draft"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ABTestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AnnotationType(Enum):
    DEFECT_TYPE_CORRECTION = "defect_type_correction"
    BBOX_ADJUSTMENT = "bbox_adjustment"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


class RetrainTriggerStatus(Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    version_tag: str
    file_path: str
    file_size_mb: float
    algorithm_type: AlgorithmType
    status: ModelVersionStatus
    description: str
    uploaded_by: str
    created_at: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    canary_lines: List[str] = field(default_factory=list)
    canary_traffic_percent: float = 0.0
    parent_version_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, model_name: str, version_tag: str, file_path: str,
               file_size_mb: float, algorithm_type: AlgorithmType,
               uploaded_by: str = "system",
               description: str = "",
               status: ModelVersionStatus = ModelVersionStatus.DRAFT,
               parent_version_id: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> "ModelVersion":
        return cls(
            version_id=str(uuid.uuid4()),
            model_name=model_name,
            version_tag=version_tag,
            file_path=file_path,
            file_size_mb=file_size_mb,
            algorithm_type=algorithm_type,
            status=status,
            description=description,
            uploaded_by=uploaded_by,
            created_at=time.time(),
            parent_version_id=parent_version_id,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "version_tag": self.version_tag,
            "file_path": self.file_path,
            "file_size_mb": self.file_size_mb,
            "algorithm_type": self.algorithm_type.value,
            "status": self.status.value,
            "description": self.description,
            "uploaded_by": self.uploaded_by,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at).isoformat() if self.created_at else "",
            "metrics": self.metrics,
            "canary_lines": self.canary_lines,
            "canary_traffic_percent": self.canary_traffic_percent,
            "parent_version_id": self.parent_version_id,
            "metadata": self.metadata
        }


@dataclass
class AnnotationRecord:
    annotation_id: str
    detection_id: str
    image_path: str
    annotation_type: AnnotationType
    original_defect_type: str
    corrected_defect_type: str
    original_bbox: Dict[str, float]
    corrected_bbox: Dict[str, float]
    annotator: str
    timestamp: float
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, detection_id: str, image_path: str,
               annotation_type: AnnotationType,
               original_defect_type: str = "",
               corrected_defect_type: str = "",
               original_bbox: Optional[Dict[str, float]] = None,
               corrected_bbox: Optional[Dict[str, float]] = None,
               annotator: str = "admin",
               notes: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> "AnnotationRecord":
        return cls(
            annotation_id=str(uuid.uuid4()),
            detection_id=detection_id,
            image_path=image_path,
            annotation_type=annotation_type,
            original_defect_type=original_defect_type,
            corrected_defect_type=corrected_defect_type,
            original_bbox=original_bbox or {},
            corrected_bbox=corrected_bbox or {},
            annotator=annotator,
            timestamp=time.time(),
            notes=notes,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "detection_id": self.detection_id,
            "image_path": self.image_path,
            "annotation_type": self.annotation_type.value,
            "original_defect_type": self.original_defect_type,
            "corrected_defect_type": self.corrected_defect_type,
            "original_bbox": self.original_bbox,
            "corrected_bbox": self.corrected_bbox,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat() if self.timestamp else "",
            "notes": self.notes,
            "metadata": self.metadata
        }


@dataclass
class ABTestConfig:
    test_id: str
    name: str
    model_a_version_id: str
    model_b_version_id: str
    status: ABTestStatus
    traffic_split_percent: float
    target_lines: List[str]
    started_at: float
    ended_at: float
    metrics_a: Dict[str, Any] = field(default_factory=dict)
    metrics_b: Dict[str, Any] = field(default_factory=dict)
    sample_count_a: int = 0
    sample_count_b: int = 0
    created_by: str = "admin"
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    winner: str = ""
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, model_a_version_id: str, model_b_version_id: str,
               traffic_split_percent: float = 50.0,
               target_lines: Optional[List[str]] = None,
               created_by: str = "admin",
               min_sample_size: int = 1000,
               confidence_level: float = 0.95,
               notes: str = "") -> "ABTestConfig":
        return cls(
            test_id=str(uuid.uuid4()),
            name=name,
            model_a_version_id=model_a_version_id,
            model_b_version_id=model_b_version_id,
            status=ABTestStatus.PENDING,
            traffic_split_percent=traffic_split_percent,
            target_lines=target_lines or [],
            started_at=0.0,
            ended_at=0.0,
            created_by=created_by,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            notes=notes
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "model_a_version_id": self.model_a_version_id,
            "model_b_version_id": self.model_b_version_id,
            "status": self.status.value,
            "traffic_split_percent": self.traffic_split_percent,
            "target_lines": self.target_lines,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
            "sample_count_a": self.sample_count_a,
            "sample_count_b": self.sample_count_b,
            "created_by": self.created_by,
            "min_sample_size": self.min_sample_size,
            "confidence_level": self.confidence_level,
            "winner": self.winner,
            "notes": self.notes,
            "metadata": self.metadata
        }


@dataclass
class RetrainTrigger:
    trigger_id: str
    trigger_reason: str
    status: RetrainTriggerStatus
    threshold_type: str
    threshold_value: float
    actual_value: float
    consecutive_days: int
    product_id: str
    algorithm_type: AlgorithmType
    sample_count: int
    started_at: float
    completed_at: float
    new_model_version_id: str = ""
    collection_dir: str = ""
    training_log: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, trigger_reason: str, threshold_type: str,
               threshold_value: float, actual_value: float,
               consecutive_days: int, product_id: str,
               algorithm_type: AlgorithmType,
               sample_count: int = 0,
               collection_dir: str = "") -> "RetrainTrigger":
        return cls(
            trigger_id=str(uuid.uuid4()),
            trigger_reason=trigger_reason,
            status=RetrainTriggerStatus.PENDING,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            actual_value=actual_value,
            consecutive_days=consecutive_days,
            product_id=product_id,
            algorithm_type=algorithm_type,
            sample_count=sample_count,
            started_at=time.time(),
            completed_at=0.0,
            collection_dir=collection_dir
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "trigger_reason": self.trigger_reason,
            "status": self.status.value,
            "threshold_type": self.threshold_type,
            "threshold_value": self.threshold_value,
            "actual_value": self.actual_value,
            "consecutive_days": self.consecutive_days,
            "product_id": self.product_id,
            "algorithm_type": self.algorithm_type.value,
            "sample_count": self.sample_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "new_model_version_id": self.new_model_version_id,
            "collection_dir": self.collection_dir,
            "training_log": self.training_log,
            "metrics": self.metrics,
            "metadata": self.metadata
        }
