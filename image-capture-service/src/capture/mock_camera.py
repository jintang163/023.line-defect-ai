import time
import numpy as np
import cv2
from typing import Optional
from src.capture.base_camera import BaseCamera
from src.utils.schemas import CameraStatus, CapturedImage, CameraConfig
from src.utils.logger import Logger

logger = Logger().logger


class MockCamera(BaseCamera):
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._frame_counter = 0
        self._mock_temperature = 25.0

    def connect(self) -> bool:
        try:
            self._update_status(CameraStatus.INITIALIZING)
            time.sleep(0.5)
            self._is_connected = True
            self._update_status(CameraStatus.ONLINE)
            logger.info(f"Mock camera {self.config.id} connected successfully")
            return True
        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            logger.error(f"Failed to connect mock camera {self.config.id}: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            self._is_connected = False
            self._update_status(CameraStatus.OFFLINE)
            logger.info(f"Mock camera {self.config.id} disconnected")
            return True
        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            return False

    def capture(self, sequence_id: str = "") -> Optional[CapturedImage]:
        if not self._is_connected:
            self._update_status(CameraStatus.ERROR, "Camera not connected")
            return None

        try:
            self._update_status(CameraStatus.CAPTURING)

            width, height = self.config.width, self.config.height
            image = self._generate_mock_image(width, height)

            self._increment_trigger_count()
            self._frame_counter += 1
            self._mock_temperature += np.random.uniform(-0.1, 0.1)

            captured = CapturedImage.create(
                camera_id=self.config.id,
                camera_position=self.config.position,
                raw_data=image,
                width=width,
                height=height,
                pixel_format=self.config.pixel_format,
                trigger_count=self._trigger_count,
                sequence_id=sequence_id,
                metadata={
                    "temperature": self._mock_temperature,
                    "frame_counter": self._frame_counter,
                    "exposure_time": self.config.exposure_time,
                    "gain": self.config.gain
                }
            )

            self._update_status(CameraStatus.ONLINE)
            logger.debug(f"Mock camera {self.config.id} captured image {captured.image_id}")
            return captured

        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            logger.error(f"Capture failed for camera {self.config.id}: {e}")
            return None

    def _generate_mock_image(self, width: int, height: int) -> np.ndarray:
        base_image = np.random.normal(128, 30, (height, width)).astype(np.uint8)

        pattern = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(pattern, f"{self.config.id}-{self._frame_counter}",
                    (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

        for _ in range(np.random.randint(0, 5)):
            x = np.random.randint(0, width - 50)
            y = np.random.randint(0, height - 50)
            w, h = np.random.randint(10, 50, 2)
            if np.random.random() > 0.5:
                cv2.rectangle(pattern, (x, y), (x + w, y + h), 200, -1)
            else:
                cv2.circle(pattern, (x + w // 2, y + h // 2), w // 2, 180, -1)

        if np.random.random() > 0.7:
            scratch_start = (np.random.randint(0, width), np.random.randint(0, height))
            scratch_end = (np.random.randint(0, width), np.random.randint(0, height))
            cv2.line(pattern, scratch_start, scratch_end, 50, np.random.randint(1, 4))

        image = cv2.addWeighted(base_image, 0.6, pattern, 0.4, 0)

        if self.config.pixel_format.startswith("Bayer"):
            bayer = np.zeros((height, width), dtype=np.uint8)
            bayer[0::2, 0::2] = image[0::2, 0::2]
            bayer[0::2, 1::2] = image[0::2, 1::2]
            bayer[1::2, 0::2] = image[1::2, 0::2]
            bayer[1::2, 1::2] = image[1::2, 1::2]
            return bayer

        return image

    def set_exposure_time(self, exposure_time: int) -> bool:
        if not self._is_connected:
            return False
        self.config.exposure_time = exposure_time
        logger.info(f"Camera {self.config.id} exposure time set to {exposure_time}us")
        return True

    def set_gain(self, gain: float) -> bool:
        if not self._is_connected:
            return False
        self.config.gain = gain
        logger.info(f"Camera {self.config.id} gain set to {gain}")
        return True

    def software_trigger(self) -> bool:
        if not self._is_connected:
            return False
        logger.debug(f"Software trigger for camera {self.config.id}")
        return True
