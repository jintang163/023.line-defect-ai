from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import threading
import json
from collections import deque
from datetime import datetime

from src.utils.schemas import (
    Defect, DefectSeverity, DefectType, AlertAction,
    DetectionOutput, ProductConfig, AlertMessage
)
from src.utils.logger import Logger

logger = Logger("alert_manager", "INFO", "./logs/defect-detection.log").logger

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

try:
    from src.notification.dispatcher import NotificationDispatcher
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertGrade(Enum):
    URGENT = "urgent"
    NORMAL = "normal"


class RelayState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class AlertManager:
    def __init__(self, max_history: int = 1000,
                 plc_connector: Optional["PLCConnector"] = None,
                 action_logger: Optional["ActionLogger"] = None,
                 consecutive_ng_threshold: int = 5,
                 auto_stop_line: bool = True,
                 notification_dispatcher: Optional["NotificationDispatcher"] = None,
                 alert_config: Optional[Dict[str, Any]] = None):
        self._alert_history: deque = deque(maxlen=max_history)
        self._alert_callbacks: Dict[AlertAction, List[Callable[[AlertMessage], None]]] = {}
        self._stats: Dict[str, int] = {
            "total_alerts": 0,
            "critical_alerts": 0,
            "reject_actions": 0,
            "stop_line_actions": 0,
            "alarm_actions": 0,
            "plc_commands_sent": 0,
            "plc_commands_failed": 0,
            "notifications_sent": 0,
            "notifications_failed": 0,
            "urgent_alerts": 0,
            "normal_alerts": 0
        }
        self._consecutive_ng_count = 0
        self._consecutive_ng_threshold = consecutive_ng_threshold
        self._auto_stop_line = auto_stop_line
        self._stop_line_active = False
        self._lock = threading.RLock()

        self._plc_connector = plc_connector
        self._action_logger = action_logger
        self._notification_dispatcher = notification_dispatcher

        self._alert_config = alert_config or {}
        self._urgent_repeat_interval = self._alert_config.get("urgent_repeat_interval_sec", 30)
        self._normal_push_enabled = self._alert_config.get("normal_push_enabled", False)

        self._relay_state = RelayState.GREEN
        self._relay_coil_map = {
            RelayState.GREEN: self._alert_config.get("relay_green_coil", 201),
            RelayState.YELLOW: self._alert_config.get("relay_yellow_coil", 202),
            RelayState.RED: self._alert_config.get("relay_red_coil", 203),
            RelayState.BUZZER: self._alert_config.get("relay_buzzer_coil", 204),
        }
        self._relay_enabled = self._alert_config.get("relay_enabled", False)

        self._pending_urgent: Dict[str, Dict[str, Any]] = {}
        self._urgent_repeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._acknowledged_alerts: set = set()

        self._alert_event_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        if self._plc_connector:
            logger.info("PLC connector integrated into alert manager")

        if self._action_logger:
            logger.info("Action logger integrated into alert manager")

        if self._notification_dispatcher and self._notification_dispatcher.enabled:
            logger.info("Notification dispatcher integrated into alert manager")

        self._register_default_callbacks()
        self._start_urgent_repeat_worker()

    def _register_default_callbacks(self):
        self._alert_callbacks[AlertAction.LOG] = [self._log_alert]
        self._alert_callbacks[AlertAction.WARN] = [self._log_alert, self._warn_alert]
        self._alert_callbacks[AlertAction.REJECT] = [self._log_alert, self._reject_alert]
        self._alert_callbacks[AlertAction.STOP_LINE] = [self._log_alert, self._stop_line_alert]
        self._alert_callbacks[AlertAction.NONE] = []

    def _start_urgent_repeat_worker(self):
        if self._urgent_repeat_thread and self._urgent_repeat_thread.is_alive():
            return
        self._urgent_repeat_thread = threading.Thread(target=self._urgent_repeat_loop, daemon=True)
        self._urgent_repeat_thread.start()

    def _urgent_repeat_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    now = time.time()
                    to_remove = []
                    for alert_id, info in self._pending_urgent.items():
                        if alert_id in self._acknowledged_alerts:
                            to_remove.append(alert_id)
                            continue
                        elapsed = now - info.get("last_notified", 0)
                        if elapsed >= self._urgent_repeat_interval:
                            self._do_notify_alert(info["alert"])
                            info["last_notified"] = now
                            self._stats["notifications_sent"] += 1
                    for aid in to_remove:
                        del self._pending_urgent[aid]
            except Exception as e:
                logger.error(f"Urgent repeat loop error: {e}", exc_info=True)

            self._stop_event.wait(5)

    def process_detection_result(self, detection_output: DetectionOutput,
                                 product_config: ProductConfig) -> List[AlertMessage]:
        with self._lock:
            alerts: List[AlertMessage] = []

            if detection_output.result.value == "OK":
                self._consecutive_ng_count = 0
                self._set_relay_state(RelayState.GREEN)
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
                logger.critical(f"Consecutive NG protection triggered: {self._consecutive_ng_count}/{self._consecutive_ng_threshold}")

            if detection_output.alert_action == AlertAction.STOP_LINE:
                self._stop_line_active = True

            for alert in alerts:
                self._alert_history.append(alert)
                self._update_stats(alert)
                grade = self._grade_alert(alert)
                self._handle_alert_notification(alert, grade)
                self._handle_relay_control(alert, grade)
                self._fire_event_callback(alert, grade)

            return alerts

    def _grade_alert(self, alert: AlertMessage) -> AlertGrade:
        if alert.level in (AlertLevel.CRITICAL.value, AlertLevel.ERROR.value):
            if alert.action == AlertAction.STOP_LINE:
                return AlertGrade.URGENT
            if alert.category == "consecutive_ng":
                return AlertGrade.URGENT
        if alert.level == AlertLevel.ERROR.value and alert.action == AlertAction.REJECT:
            if self._consecutive_ng_count >= max(2, self._consecutive_ng_threshold // 2):
                return AlertGrade.URGENT
        return AlertGrade.NORMAL

    def _handle_alert_notification(self, alert: AlertMessage, grade: AlertGrade):
        if not self._notification_dispatcher or not self._notification_dispatcher.enabled:
            return

        if grade == AlertGrade.URGENT:
            self._pending_urgent[alert.alert_id] = {
                "alert": alert,
                "grade": grade.value,
                "last_notified": time.time()
            }
            self._do_notify_alert(alert)
            self._stats["notifications_sent"] += 1
        elif grade == AlertGrade.NORMAL:
            if self._normal_push_enabled:
                self._do_notify_alert(alert)
                self._stats["notifications_sent"] += 1
            else:
                self._stats["normal_alerts"] += 1
                logger.info(f"Normal alert logged only (push suppressed): {alert.message}")

    def _do_notify_alert(self, alert: AlertMessage):
        if not self._notification_dispatcher or not self._notification_dispatcher.enabled:
            return
        try:
            self._notification_dispatcher.dispatch(
                alert_level=alert.level,
                category=alert.category,
                subject=f"[{alert.level.upper()}] {alert.category}",
                content=alert.message,
                details=alert.details
            )
        except Exception as e:
            logger.error(f"Failed to dispatch notification: {e}", exc_info=True)
            self._stats["notifications_failed"] += 1

    def _handle_relay_control(self, alert: AlertMessage, grade: AlertGrade):
        if not self._relay_enabled or not self._plc_connector or not self._plc_connector.enabled:
            return

        if grade == AlertGrade.URGENT:
            if alert.action == AlertAction.STOP_LINE:
                self._set_relay_state(RelayState.RED)
                self._activate_buzzer(True)
            else:
                self._set_relay_state(RelayState.YELLOW)
                self._activate_buzzer(True, duration_ms=2000)
        elif grade == AlertGrade.NORMAL:
            if alert.action == AlertAction.REJECT:
                self._set_relay_state(RelayState.YELLOW)
            else:
                if self._relay_state != RelayState.RED:
                    self._set_relay_state(RelayState.YELLOW)

    def _set_relay_state(self, state: RelayState):
        if not self._relay_enabled or not self._plc_connector or not self._plc_connector.enabled:
            return

        old_state = self._relay_state
        self._relay_state = state

        if old_state == state:
            return

        try:
            coil = self._relay_coil_map.get(state)
            if coil is not None:
                self._plc_connector.send_alarm_command(
                    detection_id="",
                    alarm_type=f"relay_{state.value}"
                )
                logger.info(f"Relay state changed: {old_state.value} -> {state.value}")
        except Exception as e:
            logger.error(f"Failed to set relay state to {state.value}: {e}")

    def _activate_buzzer(self, active: bool, duration_ms: int = 0):
        if not self._relay_enabled or not self._plc_connector or not self._plc_connector.enabled:
            return

        try:
            buzzer_coil = self._relay_coil_map.get(RelayState.BUZZER)
            if buzzer_coil is not None:
                if active:
                    self._plc_connector.send_alarm_command(
                        detection_id="",
                        alarm_type="buzzer_on"
                    )
                    if duration_ms > 0:
                        def _deactivate():
                            time.sleep(duration_ms / 1000.0)
                            self._activate_buzzer(False)
                        threading.Thread(target=_deactivate, daemon=True).start()
                else:
                    self._plc_connector.send_alarm_command(
                        detection_id="",
                        alarm_type="buzzer_off"
                    )
        except Exception as e:
            logger.error(f"Failed to control buzzer: {e}")

    def acknowledge_alert(self, alert_id: str, operator: str = "system") -> bool:
        with self._lock:
            self._acknowledged_alerts.add(alert_id)
            if alert_id in self._pending_urgent:
                del self._pending_urgent[alert_id]
                logger.info(f"Urgent alert acknowledged: {alert_id} by {operator}")

            for alert in self._alert_history:
                if alert.alert_id == alert_id:
                    alert.details["acknowledged"] = True
                    alert.details["acknowledged_by"] = operator
                    alert.details["acknowledged_at"] = time.time()
                    break

            self._fire_event_callback({
                "event": "acknowledge",
                "alert_id": alert_id,
                "operator": operator,
                "timestamp": time.time()
            })

            if self._relay_state == RelayState.RED and not self._pending_urgent:
                self._set_relay_state(RelayState.YELLOW)
                self._activate_buzzer(False)

            return True

    def get_alert_history(self, limit: int = 100, level: Optional[str] = None,
                          category: Optional[str] = None, grade: Optional[str] = None,
                          acknowledged: Optional[bool] = None,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        with self._lock:
            alerts = list(self._alert_history)

            if level:
                alerts = [a for a in alerts if a.level == level]
            if category:
                alerts = [a for a in alerts if a.category == category]
            if grade:
                alerts = [a for a in alerts if self._grade_alert(a).value == grade]
            if acknowledged is not None:
                alerts = [a for a in alerts if a.details.get("acknowledged", False) == acknowledged]
            if start_time:
                alerts = [a for a in alerts if a.timestamp >= start_time]
            if end_time:
                alerts = [a for a in alerts if a.timestamp <= end_time]

            alerts = alerts[-limit:]
            return [a.to_dict() for a in alerts]

    def get_pending_urgent_alerts(self) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for alert_id, info in self._pending_urgent.items():
                alert_dict = info["alert"].to_dict()
                alert_dict["grade"] = info["grade"]
                alert_dict["last_notified"] = info["last_notified"]
                alert_dict["repeat_count"] = alert_dict.get("details", {}).get("repeat_count", 0)
                result.append(alert_dict)
            return result

    def get_relay_state(self) -> str:
        return self._relay_state.value

    def register_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        with self._lock:
            self._alert_event_callbacks.append(callback)

    def _fire_event_callback(self, event_data: Any):
        for cb in self._alert_event_callbacks:
            try:
                cb(event_data)
            except Exception as e:
                logger.error(f"Error in alert event callback: {e}", exc_info=True)

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
        logger.warning(f"Warning alert triggered: {alert.message}")

        if self._plc_connector and self._plc_connector.enabled:
            try:
                self._plc_connector.send_alarm_command(
                    detection_id=alert.detection_id,
                    alarm_type=alert.category
                )
                self._stats["plc_commands_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send alarm PLC command: {e}")
                self._stats["plc_commands_failed"] += 1

        if self._action_logger and self._action_logger.enabled:
            try:
                self._action_logger.log_alert(alert)
            except Exception as e:
                logger.warning(f"Failed to log alert action: {e}")

    def _reject_alert(self, alert: AlertMessage):
        self._stats["reject_actions"] += 1
        logger.warning(f"Reject signal sent: {alert.message}")

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
                logger.error(f"Failed to send reject PLC command: {e}")
                self._stats["plc_commands_failed"] += 1

        if self._action_logger and self._action_logger.enabled:
            try:
                self._action_logger.log_alert(alert)
            except Exception as e:
                logger.warning(f"Failed to log reject action: {e}")

    def _stop_line_alert(self, alert: AlertMessage):
        self._stats["stop_line_actions"] += 1
        self._stop_line_active = True
        logger.critical(f"Stop line signal sent: {alert.message}")

        if self._plc_connector and self._plc_connector.enabled:
            try:
                self._plc_connector.send_stop_line_command(
                    detection_id=alert.detection_id,
                    reason=alert.message
                )
                self._stats["plc_commands_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send stop line PLC command: {e}")
                self._stats["plc_commands_failed"] += 1

        if self._action_logger and self._action_logger.enabled:
            try:
                self._action_logger.log_alert(alert)
            except Exception as e:
                logger.warning(f"Failed to log stop line action: {e}")

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
            logger.info(f"Stop line reset | operator: {operator} | reason: {reason}")

            if self._plc_connector and self._plc_connector.enabled:
                try:
                    self._plc_connector.send_reset_command()
                    self._stats["plc_commands_sent"] += 1
                    logger.info("PLC reset command sent")
                except Exception as e:
                    logger.error(f"Failed to send PLC reset command: {e}")
                    self._stats["plc_commands_failed"] += 1

            self._set_relay_state(RelayState.GREEN)
            self._activate_buzzer(False)

            if self._action_logger and self._action_logger.enabled:
                try:
                    from src.utils.schemas import ActionLogType
                    self._action_logger.log_system_event(
                        event=f"Stop line reset | operator: {operator} | reason: {reason}",
                        level="info",
                        source="alert_manager",
                        details={"operator": operator, "reason": reason}
                    )
                except Exception as e:
                    logger.warning(f"Failed to log reset action: {e}")

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
                "action_logger_enabled": self._action_logger.enabled if self._action_logger else False,
                "relay_state": self._relay_state.value,
                "pending_urgent_count": len(self._pending_urgent),
                "notification_enabled": self._notification_dispatcher.enabled if self._notification_dispatcher else False,
                "relay_enabled": self._relay_enabled
            }

    def clear_history(self):
        with self._lock:
            self._alert_history.clear()
            self._stats = {k: 0 for k in self._stats}
            self._consecutive_ng_count = 0
            logger.info("Alert history cleared")

    @property
    def stop_line_active(self) -> bool:
        return self._stop_line_active

    @property
    def consecutive_ng_count(self) -> int:
        return self._consecutive_ng_count

    def set_consecutive_ng_threshold(self, threshold: int):
        self._consecutive_ng_threshold = max(1, threshold)
        logger.info(f"Consecutive NG threshold set to: {self._consecutive_ng_threshold}")

    def shutdown(self):
        self._stop_event.set()
        if self._urgent_repeat_thread and self._urgent_repeat_thread.is_alive():
            self._urgent_repeat_thread.join(timeout=5)
        logger.info("Alert manager shutdown complete")
