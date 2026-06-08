import threading
from collections import deque
from typing import Dict, List, Optional, Any, Callable

from src.utils.schemas import (
    ManualOverrideRecord,
    ManualOverrideAction,
    DetectionResult,
    DetectionOutput,
)
from src.utils.logger import Logger

try:
    from src.plc.plc_connector import PLCConnector
    PLC_AVAILABLE = True
except ImportError:
    PLC_AVAILABLE = False


class ManualOverrideManager:
    def __init__(self, max_history: int = 10000, plc_connector: Optional["PLCConnector"] = None):
        self._max_history = max_history
        self._history: deque[ManualOverrideRecord] = deque(maxlen=max_history)
        self._sequence_map: Dict[str, ManualOverrideRecord] = {}
        self._callbacks: List[Callable[[ManualOverrideRecord], None]] = []
        self._lock = threading.Lock()
        self._logger = Logger()
        self._plc_connector = plc_connector

        if self._plc_connector:
            self._logger.info("✅ PLC连接器已集成到人工干预管理器")

    def apply_override(
        self,
        sequence_id: str,
        detection_id: str,
        action: ManualOverrideAction,
        operator: str,
        reason: str,
        original_result: DetectionResult,
        final_result: DetectionResult,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[ManualOverrideRecord]:
        if not sequence_id or not detection_id or not operator or not reason:
            self._logger.error(
                "Invalid override parameters: sequence_id, detection_id, operator, and reason are required"
            )
            return None

        if action not in (
            ManualOverrideAction.FORCE_PASS,
            ManualOverrideAction.FORCE_REJECT,
        ):
            self._logger.error(f"Invalid override action: {action}")
            return None

        with self._lock:
            record = ManualOverrideRecord.create(
                sequence_id=sequence_id,
                detection_id=detection_id,
                action=action,
                operator=operator,
                reason=reason,
                original_result=original_result,
                final_result=final_result,
                details=details,
            )

            self._history.append(record)
            self._sequence_map[sequence_id] = record

            self._logger.info(
                f"🔧 人工干预已应用: sequence_id={sequence_id}, "
                f"detection_id={detection_id}, "
                f"action={action.value}, operator={operator}, "
                f"original={original_result.value}, final={final_result.value}"
            )

            self._trigger_plc_action(record)

            for callback in self._callbacks:
                try:
                    callback(record)
                except Exception as e:
                    self._logger.error(f"Error in override callback: {e}")

            return record

    def _trigger_plc_action(self, record: ManualOverrideRecord):
        if not self._plc_connector or not self._plc_connector.enabled:
            return

        try:
            from src.utils.schemas import AlertAction

            if record.action == ManualOverrideAction.FORCE_REJECT:
                self._plc_connector.send_reject_command(
                    detection_id=record.detection_id,
                    defect_types=None,
                    alert_action=AlertAction.REJECT
                )
                self._logger.info(f"📤 人工剔除PLC指令已发送: sequence_id={record.sequence_id}")

            elif record.action == ManualOverrideAction.FORCE_PASS:
                self._logger.info(f"✅ 人工放行: sequence_id={record.sequence_id}，不发送剔除信号")

        except Exception as e:
            self._logger.error(f"发送人工干预PLC指令失败: {e}", exc_info=True)

    def get_override(self, sequence_id: str) -> Optional[ManualOverrideRecord]:
        with self._lock:
            return self._sequence_map.get(sequence_id)

    def get_override_by_detection_id(self, detection_id: str) -> Optional[ManualOverrideRecord]:
        with self._lock:
            for record in self._history:
                if record.detection_id == detection_id:
                    return record
            return None

    def get_overrides(
        self,
        operator: Optional[str] = None,
        action: Optional[ManualOverrideAction] = None,
        limit: int = 100,
    ) -> List[ManualOverrideRecord]:
        with self._lock:
            results = list(self._history)

            if operator:
                results = [r for r in results if r.operator == operator]

            if action:
                results = [r for r in results if r.action == action]

            results = sorted(results, key=lambda r: r.timestamp, reverse=True)

            return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._history)

            if total == 0:
                return {
                    "total_overrides": 0,
                    "force_pass_count": 0,
                    "force_reject_count": 0,
                    "pass_rate": 0.0,
                    "reject_rate": 0.0,
                    "unique_operators": 0,
                    "operator_distribution": {},
                    "action_distribution": {},
                    "original_result_distribution": {},
                    "final_result_distribution": {},
                }

            force_pass_count = sum(
                1 for r in self._history if r.action == ManualOverrideAction.FORCE_PASS
            )
            force_reject_count = sum(
                1 for r in self._history if r.action == ManualOverrideAction.FORCE_REJECT
            )

            operator_counts: Dict[str, int] = {}
            for r in self._history:
                operator_counts[r.operator] = operator_counts.get(r.operator, 0) + 1

            original_result_counts: Dict[str, int] = {}
            for r in self._history:
                key = r.original_result.value
                original_result_counts[key] = original_result_counts.get(key, 0) + 1

            final_result_counts: Dict[str, int] = {}
            for r in self._history:
                key = r.final_result.value
                final_result_counts[key] = final_result_counts.get(key, 0) + 1

            return {
                "total_overrides": total,
                "force_pass_count": force_pass_count,
                "force_reject_count": force_reject_count,
                "pass_rate": force_pass_count / total,
                "reject_rate": force_reject_count / total,
                "unique_operators": len(operator_counts),
                "operator_distribution": operator_counts,
                "action_distribution": {
                    ManualOverrideAction.FORCE_PASS.value: force_pass_count,
                    ManualOverrideAction.FORCE_REJECT.value: force_reject_count,
                },
                "original_result_distribution": original_result_counts,
                "final_result_distribution": final_result_counts,
            }

    def register_override_callback(
        self, callback: Callable[[ManualOverrideRecord], None]
    ) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                self._logger.info("Override callback registered")

    def has_override(self, sequence_id: str) -> bool:
        with self._lock:
            return sequence_id in self._sequence_map

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._sequence_map.clear()
            self._logger.info("Manual override history cleared")
