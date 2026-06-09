from typing import Dict, Any, Optional, List
import threading
import time
import json
import base64
import numpy as np

from src.utils.logger import Logger

from src.system_monitor.role_manager import RoleManager
from src.system_monitor.param_adjuster import ParamAdjuster
from src.system_monitor.health_checker import HealthChecker

logger = Logger("system_monitor", "INFO", "./logs/defect-detection.log").logger


class SystemMonitorManager:
    def __init__(self, config: Dict[str, Any],
                 algorithm_manager=None,
                 config_manager=None,
                 message_consumer=None,
                 result_producer=None,
                 data_management_manager=None):
        self._config = config
        self._enabled = config.get("enable", False)
        self._algorithm_manager = algorithm_manager
        self._config_manager = config_manager
        self._message_consumer = message_consumer
        self._result_producer = result_producer
        self._data_management_manager = data_management_manager

        self._role_manager: Optional[RoleManager] = None
        self._param_adjuster: Optional[ParamAdjuster] = None
        self._health_checker: Optional[HealthChecker] = None

        self._latest_annotated_frames: Dict[str, bytes] = {}
        self._frame_lock = threading.Lock()
        self._frame_subscribers: List = []

        if not self._enabled:
            logger.info("System monitor module is disabled")
            return

        self._init_components()

    def _init_components(self):
        auth_config = self._config.get("auth", {})
        self._role_manager = RoleManager(auth_config)
        logger.info("Role manager initialized")

        self._param_adjuster = ParamAdjuster(
            self._algorithm_manager,
            self._config_manager
        )
        logger.info("Param adjuster initialized")

        health_config = self._config.get("health_check", {})
        self._health_checker = HealthChecker(
            config=health_config,
            message_consumer=self._message_consumer,
            result_producer=self._result_producer,
            algorithm_manager=self._algorithm_manager
        )
        logger.info("Health checker initialized")

    def start(self):
        if not self._enabled:
            return
        if self._health_checker:
            self._health_checker.start()
        logger.info("System monitor manager started")

    def stop(self):
        if self._health_checker:
            self._health_checker.stop()
        logger.info("System monitor manager stopped")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def role_manager(self) -> Optional[RoleManager]:
        return self._role_manager

    @property
    def param_adjuster(self) -> Optional[ParamAdjuster]:
        return self._param_adjuster

    @property
    def health_checker(self) -> Optional[HealthChecker]:
        return self._health_checker

    def authenticate(self, username: str, password: str) -> Optional[str]:
        if self._role_manager:
            return self._role_manager.authenticate(username, password)
        return None

    def verify_token(self, token: str) -> Optional[Dict]:
        if self._role_manager:
            return self._role_manager.verify_token(token)
        return None

    def check_permission(self, token: str, permission: str) -> bool:
        if self._role_manager:
            return self._role_manager.check_permission(token, permission)
        return False

    def update_annotated_frame(self, camera_id: str, frame_data: bytes):
        with self._frame_lock:
            self._latest_annotated_frames[camera_id] = frame_data

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_annotated_frames.get(camera_id)

    def get_all_camera_ids(self) -> List[str]:
        with self._frame_lock:
            return list(self._latest_annotated_frames.keys())

    def get_health_status(self) -> Dict:
        if self._health_checker:
            return self._health_checker.check_all()
        return {}

    def get_history_dashboard_data(self) -> Dict[str, Any]:
        result = {
            "yield_trend": [],
            "defect_distribution": [],
            "avg_processing_time": []
        }

        if self._data_management_manager and self._data_management_manager.enabled:
            try:
                now = time.time()
                start_24h = now - 86400

                yield_data = self._data_management_manager.get_yield_trend(
                    start_time=start_24h, end_time=now, interval="hour"
                )
                result["yield_trend"] = yield_data.get("trend", [])

                defect_data = self._data_management_manager.get_defect_distribution(
                    start_time=start_24h, end_time=now
                )
                result["defect_distribution"] = defect_data.get("distribution", [])

                overview = self._data_management_manager.get_overview(
                    start_time=start_24h, end_time=now
                )
                result["overview"] = overview
            except Exception as e:
                logger.error(f"Failed to get dashboard data: {e}")

        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "auth_enabled": self._role_manager is not None,
            "health_check_enabled": self._health_checker is not None,
            "camera_feeds": len(self.get_all_camera_ids())
        }
