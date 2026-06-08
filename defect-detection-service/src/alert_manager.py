from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import threading
from collections import deque

from src.utils.schemas import (
    Defect, DefectSeverity, DefectType, AlertAction,
    DetectionOutput, ProductConfig, AlertMessage
)
from src.utils.logger import Logger

logger = Logger().logger

try:
    from src.plc.plc_connector import PLCConnector
    PLC_AVAILABLE = True
except ImportError:
    PLC_AVAILABLE = False

try:
    from src.action_logger.action_logger import ActionLogger
    ACTION_LOGGER_AVAILABLE = True
except ImportError:
    ACTION_LOGGER_AVAILABLE = False


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertManager:
    def __init__(self, max_history: int = 1000,
                 plc_connector: Optional["PLCConnector"] = None,
                 action_logger: Optional["ActionLogger"] = None,
                 consecutive_ng_threshold: int = 5,
                 auto_stop_line: bool = True):
        self._alert_history: deque = deque(maxlen=max_history)
        self._alert_callbacks: Dict[AlertAction, List[Callable[[AlertMessage], None]]] = {}
        self._stats: Dict[str, int] = {
            "total_alerts": 0,
            "critical_alerts": 0,
            "reject_actions": 0,
            "stop_line_actions": 0,
            "alarm_actions": 0,
            "plc_commands_sent": 0,
            "plc_commands_failed": 0
        }
        self._consecutive_ng_count = 0
        self._consecutive_ng_threshold = consecutive_ng_threshold
        self._auto_stop_line = auto_stop_line
        self._stop_line_active = False
        self._lock = threading.RLock()

        self._plc_connector = plc_connector
        self._action_logger = action_logger

        if self._plc_connector:
            logger.info("✅ PLC连接器已集成到告警管理器")

        if self._action_logger:
            logger.info("✅ 动作日志记录器已集成到告警管理器")

        self._register_default_callbacks()

    def _register_default_callbacks(self):
        self._alert_callbacks[AlertAction.LOG] = [self._log_alert]
        self._alert_callbacks[AlertAction.WARN] = [self._log_alert, self._warn_alert]
        self._alert_callbacks[AlertAction.REJECT] = [self._log_alert, self._reject_alert]
        self._alert_callbacks[AlertAction.STOP_LINE] = [self._log_alert, self._stop_line_alert]
        self._alert_callbacks[AlertAction.NONE] = []

    def process_detection_result(self, detection_output: DetectionOutput,
                                 product_config: ProductConfig) -> List[AlertMessage]:
        with self._lock:
            alerts: List[AlertMessage] = []

            if detection_output.result.value == "OK":
                self._consecutive_ng_count = 0
                return alerts

            self._consecutive_ng_count += 1

            for defect in detection_output.defects:
                alert = self._create_alert_for_defect(
                    defect, detection_output, product_config
                )
                if alert:
                    alerts.append(alert)
                    self._execute_alert_callbacks(alert)

            if self._auto_stop_line and self._consecutive_ng_count >= self._consecutive_ng_threshold:
                stop_alert = AlertMessage.create(
                    level=AlertLevel.CRITICAL.value,
                    category="consecutive_ng",
                    message=f"连续{self._consecutive_ng_count}个产品检测不合格，超过阈值{self._consecutive_ng_threshold}",
                    source="alert_manager",
                    action=AlertAction.STOP_LINE,
                    detection_id=detection_output.detection_id,
                    details={
                        "consecutive_count": self._consecutive_ng_count,
                        "threshold": self._consecutive_ng_threshold,
                        "auto_triggered": True
                    }
                )
                alerts.append(stop_alert)
                self._execute_alert_callbacks(stop_alert)
                logger.critical(f"🔴 连续NG保护触发: {self._consecutive_ng_count}/{self._consecutive_ng_threshold}，自动停机")

            if detection_output.alert_action == AlertAction.STOP_LINE:
                self._stop_line_active = True

            for alert in alerts:
                self._alert_history.append(alert)
                self._update_stats(alert)

            return alerts

    def _create_alert_for_defect(self, defect: Defect, detection_output: DetectionOutput,
                                 product_config: ProductConfig) -> Optional[AlertMessage]:
        defect_config = product_config.get_defect_config(defect.type)

        if defect_config:
            alert_action = defect_config.alert_action
        else:
            alert_action = self._get_default_alert_action(defect.severity)

        if alert_action == AlertAction.NONE:
            return None

        alert_level = self._severity_to_alert_level(defect.severity)

        category_map = {
            DefectType.SCRATCH: "surface_defect",
            DefectType.DIRT: "contamination",
            DefectType.DENT: "shape_defect",
            DefectType.CRACK: "structure_defect",
            DefectType.MISSING: "missing_part",
            DefectType.STAIN: "contamination",
            DefectType.DEFORMATION: "shape_defect",
            DefectType.BUBBLE: "material_defect",
            DefectType.UNKNOWN: "unknown_defect"
        }

        category = category_map.get(defect.type, "unknown_defect")

        return AlertMessage.create(
            level=alert_level.value,
            category=category,
            message=f"{self._get_defect_type_label(defect.type)}: "
                    f"{defect.severity.value}, "
                    f"面积: {defect.area_mm2:.4f}mm², "
                    f"置信度: {defect.confidence:.3f}",
            source=f"defect_{defect.type.value}",
            action=alert_action,
            detection_id=detection_output.detection_id,
            defect_id=defect.defect_id,
            details={
                "defect_type": defect.type.value,
                "severity": defect.severity.value,
                "confidence": defect.confidence,
                "area_mm2": defect.area_mm2,
                "area_pixels": defect.area_pixels,
                "bbox": defect.bbox.to_dict(),
                "product_id": product_config.product_id
            }
        )

    def _get_default_alert_action(self, severity: DefectSeverity) -> AlertAction:
        action_map = {
            DefectSeverity.CRITICAL: AlertAction.STOP_LINE,
            DefectSeverity.MAJOR: AlertAction.REJECT,
            DefectSeverity.MINOR: AlertAction.LOG,
            DefectSeverity.WARNING: AlertAction.WARN
        }
        return action_map.get(severity, AlertAction.LOG)

    def _severity_to_alert_level(self, severity: DefectSeverity) -> AlertLevel:
        level_map = {
            DefectSeverity.CRITICAL: AlertLevel.CRITICAL,
            DefectSeverity.MAJOR: AlertLevel.ERROR,
            DefectSeverity.MINOR: AlertLevel.WARNING,
            DefectSeverity.WARNING: AlertLevel.INFO
        }
        return level_map.get(severity, AlertLevel.INFO)

    def _get_defect_type_label(self, defect_type: DefectType) -> str:
        label_map = {
            DefectType.SCRATCH: "划痕",
            DefectType.DIRT: "脏污",
            DefectType.DENT: "凹痕",
            DefectType.CRACK: "裂纹",
            DefectType.MISSING: "缺失",
            DefectType.STAIN: "污渍",
            DefectType.DEFORMATION: "变形",
            DefectType.BUBBLE: "气泡",
            DefectType.UNKNOWN: "未知缺陷"
        }
        return label_map.get(defect_type, defect_type.value)

    def _execute_alert_callbacks(self, alert: AlertMessage):
        callbacks = self._alert_callbacks.get(alert.action, [])
        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error executing alert callback: {e}", exc_info=True)

    def _log_alert(self, alert: AlertMessage):
        log_func = {
            AlertLevel.INFO.value: logger.info,
            AlertLevel.WARNING.value: logger.warning,
            AlertLevel.ERROR.value: logger.error,
            AlertLevel.CRITICAL.value: logger.critical
        }.get(alert.level, logger.info)

        log_func(
            f"[ALERT] {alert.level.upper()} | {alert.category} | "
            f"Action: {alert.action.value} | {alert.message}"
        )

    def _warn_alert(self, alert: AlertMessage):
        self._stats["alarm_actions"] += 1
        logger.warning(f"⚠️  警告告警触发: {alert.message}")

        if self._plc_connector and self._plc_connector.enabled:
            try:
                self._plc_connector.send_alarm_command(
                    detection_id=alert.detection_id,
                    alarm_type=alert.category
                )
                self._stats["plc_commands_sent"] += 1
            except Exception as e:
                logger.error(f"发送报警PLC指令失败: {e}")
                self._stats["plc_commands_failed"] += 1

        if self._action_logger and self._action_logger.enabled:
            try:
                self._action_logger.log_alert(alert)
            except Exception as e:
                logger.warning(f"记录告警动作日志失败: {e}")

    def _reject_alert(self, alert: AlertMessage):
        self._stats["reject_actions"] += 1
        logger.warning(f"🚫 剔除信号已发送: {alert.message}")

        if self._plc_connector and self._plc_connector.enabled:
            try:
                defect_types = []
                if "defect_type" in alert.details:
                    from src.utils.schemas import DefectType
                    try:
                        dt = DefectType(alert.details["defect_type"])
                        defect_types.append(dt)
                    except:
                        pass

                self._plc_connector.send_reject_command(
                    detection_id=alert.detection_id,
                    defect_types=defect_types,
                    alert_action=AlertAction.REJECT
                )
                self._stats["plc_commands_sent"] += 1
            except Exception as e:
                logger.error(f"发送剔除PLC指令失败: {e}")
                self._stats["plc_commands_failed"] += 1

        if self._action_logger and self._action_logger.enabled:
            try:
                self._action_logger.log_alert(alert)
            except Exception as e:
                logger.warning(f"记录剔除动作日志失败: {e}")

    def _stop_line_alert(self, alert: AlertMessage):
        self._stats["stop_line_actions"] += 1
        self._stop_line_active = True
        logger.critical(f"🛑 生产线停机信号已发送: {alert.message}")

        if self._plc_connector and self._plc_connector.enabled:
            try:
                self._plc_connector.send_stop_line_command(
                    detection_id=alert.detection_id,
                    reason=alert.message
                )
                self._stats["plc_commands_sent"] += 1
            except Exception as e:
                logger.error(f"发送停机PLC指令失败: {e}")
                self._stats["plc_commands_failed"] += 1

        if self._action_logger and self._action_logger.enabled:
            try:
                self._action_logger.log_alert(alert)
            except Exception as e:
                logger.warning(f"记录停机动作日志失败: {e}")

    def _update_stats(self, alert: AlertMessage):
        self._stats["total_alerts"] += 1
        if alert.level == AlertLevel.CRITICAL.value:
            self._stats["critical_alerts"] += 1

    def register_callback(self, action: AlertAction, callback: Callable[[AlertMessage], None]):
        with self._lock:
            if action not in self._alert_callbacks:
                self._alert_callbacks[action] = []
            self._alert_callbacks[action].append(callback)
            logger.info(f"Registered callback for action: {action.value}")

    def clear_callbacks(self, action: Optional[AlertAction] = None):
        with self._lock:
            if action:
                self._alert_callbacks[action] = []
            else:
                self._alert_callbacks = {k: [] for k in self._alert_callbacks}

    def reset_stop_line(self, operator: str = "system", reason: str = "系统重置"):
        with self._lock:
            self._stop_line_active = False
            self._consecutive_ng_count = 0
            logger.info(f"🔄 生产线停机状态已重置 | 操作员: {operator} | 原因: {reason}")

            if self._plc_connector and self._plc_connector.enabled:
                try:
                    self._plc_connector.send_reset_command()
                    self._stats["plc_commands_sent"] += 1
                    logger.info("✅ PLC复位指令已发送")
                except Exception as e:
                    logger.error(f"发送PLC复位指令失败: {e}")
                    self._stats["plc_commands_failed"] += 1

            if self._action_logger and self._action_logger.enabled:
                try:
                    from src.utils.schemas import ActionLogType
                    self._action_logger.log_system_event(
                        event=f"生产线停机状态已重置 | 操作员: {operator} | 原因: {reason}",
                        level="info",
                        source="alert_manager",
                        details={"operator": operator, "reason": reason}
                    )
                except Exception as e:
                    logger.warning(f"记录重置动作日志失败: {e}")

    def get_recent_alerts(self, limit: int = 100, level: Optional[str] = None) -> List[AlertMessage]:
        with self._lock:
            alerts = list(self._alert_history)
            if level:
                alerts = [a for a in alerts if a.level == level]
            return alerts[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "consecutive_ng_count": self._consecutive_ng_count,
                "consecutive_ng_threshold": self._consecutive_ng_threshold,
                "auto_stop_line": self._auto_stop_line,
                "stop_line_active": self._stop_line_active,
                "alert_history_size": len(self._alert_history),
                "plc_connected": self._plc_connector.is_connected() if self._plc_connector else False,
                "plc_enabled": self._plc_connector.enabled if self._plc_connector else False,
                "action_logger_enabled": self._action_logger.enabled if self._action_logger else False
            }

    def clear_history(self):
        with self._lock:
            self._alert_history.clear()
            self._stats = {k: 0 for k in self._stats}
            self._consecutive_ng_count = 0
            logger.info("告警历史已清除")

    @property
    def stop_line_active(self) -> bool:
        return self._stop_line_active

    @property
    def consecutive_ng_count(self) -> int:
        return self._consecutive_ng_count

    def set_consecutive_ng_threshold(self, threshold: int):
        self._consecutive_ng_threshold = max(1, threshold)
        logger.info(f"连续不合格阈值设置为: {self._consecutive_ng_threshold}")
