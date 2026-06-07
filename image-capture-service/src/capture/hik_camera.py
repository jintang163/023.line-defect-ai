import time
import numpy as np
from typing import Optional
from src.capture.base_camera import BaseCamera
from src.utils.schemas import CameraStatus, CapturedImage, CameraConfig
from src.utils.logger import Logger

logger = Logger().logger


class HikvisionCamera(BaseCamera):
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._sdk = None
        self._handle = None

    def _try_import_sdk(self) -> bool:
        try:
            import cv2
            self._sdk = cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available, Hikvision camera will use mock mode")
            return False

    def connect(self) -> bool:
        if not self._try_import_sdk():
            return False

        try:
            self._update_status(CameraStatus.INITIALIZING)
            time.sleep(0.3)

            self._is_connected = True
            self._update_status(CameraStatus.ONLINE)
            logger.info(f"Hikvision camera {self.config.id} connected successfully")
            return True

        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            logger.error(f"Failed to connect Hikvision camera {self.config.id}: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self._handle is not None:
                self._handle = None
            self._is_connected = False
            self._update_status(CameraStatus.OFFLINE)
            logger.info(f"Hikvision camera {self.config.id} disconnected")
            return True
        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            return False

    def capture(self, sequence_id: str = "") -> Optional[CapturedImage]:
        if not self._is_connected:
            return None

        try:
            self._update_status(CameraStatus.CAPTURING)
            image = self._generate_hik_image()

            self._increment_trigger_count()
            captured = CapturedImage.create(
                camera_id=self.config.id,
                camera_position=self.config.position,
                raw_data=image,
                width=self.config.width,
                height=self.config.height,
                pixel_format=self.config.pixel_format,
                trigger_count=self._trigger_count,
                sequence_id=sequence_id,
                metadata={
                    "exposure_time": self.config.exposure_time,
                    "gain": self.config.gain,
                    "sdk": "hikvision"
                }
            )

            self._update_status(CameraStatus.ONLINE)
            logger.debug(f"Hikvision camera {self.config.id} captured image {captured.image_id}")
            return captured

        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            logger.error(f"Capture failed for Hikvision camera {self.config.id}: {e}")
            return None

    def _generate_hik_image(self) -> np.ndarray:
        import cv2
        image = np.random.normal(130, 25, (self.config.height, self.config.width)).astype(np.uint8)
        cv2.putText(image, f"HIK-{self.config.id}", (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        return image

    def set_exposure_time(self, exposure_time: int) -> bool:
        if not self._is_connected:
            return False
        self.config.exposure_time = exposure_time
        logger.info(f"Hikvision camera {self.config.id} exposure time set to {exposure_time}us")
        return True

    def set_gain(self, gain: float) -> bool:
        if not self._is_connected:
            return False
        self.config.gain = gain
        logger.info(f"Hikvision camera {self.config.id} gain set to {gain}")
        return True

    def software_trigger(self) -> bool:
        if not self._is_connected:
            return False
        logger.debug(f"Software trigger for Hikvision camera {self.config.id}")
        return True
