from abc import ABC, abstractmethod
from typing import Optional
import time
import numpy as np
from src.utils.schemas import (
    CameraConfig, CameraStatus, CameraStatusInfo, CapturedImage
)
from src.utils.logger import Logger

logger = Logger().logger


class BaseCamera(ABC):
    def __init__(self, config: CameraConfig):
        self.config = config
        self._status = CameraStatus.OFFLINE
        self._trigger_count = 0
        self._last_capture_time: Optional[float] = None
        self._error_message: Optional[str] = None
        self._is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def capture(self, sequence_id: str = "") -> Optional[CapturedImage]:
        pass

    @abstractmethod
    def set_exposure_time(self, exposure_time: int) -> bool:
        pass

    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        pass

    @abstractmethod
    def software_trigger(self) -> bool:
        pass

    def is_connected(self) -> bool:
        return self._is_connected

    def get_status(self) -> CameraStatusInfo:
        return CameraStatusInfo(
            camera_id=self.config.id,
            status=self._status,
            exposure_time=self.config.exposure_time,
            gain=self.config.gain,
            trigger_count=self._trigger_count,
            last_capture_time=self._last_capture_time,
            error_message=self._error_message
        )

    def _update_status(self, status: CameraStatus, error_msg: Optional[str] = None):
        self._status = status
        self._error_message = error_msg
        if error_msg:
            logger.error(f"Camera {self.config.id} status changed to {status.value}: {error_msg}")
        else:
            logger.info(f"Camera {self.config.id} status changed to {status.value}")

    def _increment_trigger_count(self):
        self._trigger_count += 1
        self._last_capture_time = time.time()
