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


class ManualOverrideManager:
    def __init__(self, max_history: int = 10000):
        self._max_history = max_history
        self._history: deque[ManualOverrideRecord] = deque(maxlen=max_history)
        self._detection_map: Dict[str, ManualOverrideRecord] = {}
        self._callbacks: List[Callable[[ManualOverrideRecord], None]] = []
        self._lock = threading.Lock()
        self._logger = Logger()

    def apply_override(
        self,
        detection_id: str,
        action: ManualOverrideAction,
        operator: str,
        reason: str,
        original_result: DetectionResult,
        final_result: DetectionResult,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[ManualOverrideRecord]:
        if not detection_id or not operator or not reason:
            self._logger.error(
                "Invalid override parameters: detection_id, operator, and reason are required"
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
                detection_id=detection_id,
                action=action,
                operator=operator,
                reason=reason,
                original_result=original_result,
                final_result=final_result,
                details=details,
            )

            self._history.append(record)
            self._detection_map[detection_id] = record

            self._logger.info(
                f"Manual override applied: detection_id={detection_id}, "
                f"action={action.value}, operator={operator}, "
                f"original={original_result.value}, final={final_result.value}"
            )

            for callback in self._callbacks:
                try:
                    callback(record)
                except Exception as e:
                    self._logger.error(f"Error in override callback: {e}")

            return record

    def get_override(self, detection_id: str) -> Optional[ManualOverrideRecord]:
        with self._lock:
            return self._detection_map.get(detection_id)

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

    def has_override(self, detection_id: str) -> bool:
        with self._lock:
            return detection_id in self._detection_map

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._detection_map.clear()
            self._logger.info("Manual override history cleared")
