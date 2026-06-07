import time
import numpy as np
from typing import Optional
from src.capture.base_camera import BaseCamera
from src.utils.schemas import CameraStatus, CapturedImage, CameraConfig
from src.utils.logger import Logger

logger = Logger().logger


class BaslerCamera(BaseCamera):
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._pylon = None
        self._camera = None

    def _try_import_pylon(self) -> bool:
        try:
            from pypylon import pylon
            self._pylon = pylon
            return True
        except ImportError:
            logger.warning("pypylon not installed, Basler camera will use mock mode")
            return False

    def connect(self) -> bool:
        if not self._try_import_pylon():
            return self._mock_connect()

        try:
            self._update_status(CameraStatus.INITIALIZING)
            tl_factory = self._pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()

            target_device = None
            for dev in devices:
                if dev.GetSerialNumber() == self.config.serial_number:
                    target_device = dev
                    break

            if target_device is None:
                raise RuntimeError(f"Basler camera with serial {self.config.serial_number} not found")

            self._camera = self._pylon.InstantCamera(tl_factory.CreateDevice(target_device))
            self._camera.Open()

            self._camera.ExposureTime.SetValue(self.config.exposure_time)
            self._camera.Gain.SetValue(self.config.gain)
            self._camera.Width.SetValue(self.config.width)
            self._camera.Height.SetValue(self.config.height)
            self._camera.PixelFormat.SetValue(self.config.pixel_format)

            self._is_connected = True
            self._update_status(CameraStatus.ONLINE)
            logger.info(f"Basler camera {self.config.id} connected successfully")
            return True

        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            logger.error(f"Failed to connect Basler camera {self.config.id}: {e}")
            return False

    def _mock_connect(self) -> bool:
        self._update_status(CameraStatus.INITIALIZING)
        time.sleep(0.5)
        self._is_connected = True
        self._update_status(CameraStatus.ONLINE)
        logger.info(f"Basler camera {self.config.id} connected in mock mode")
        return True

    def disconnect(self) -> bool:
        try:
            if self._camera is not None:
                self._camera.Close()
                self._camera = None
            self._is_connected = False
            self._update_status(CameraStatus.OFFLINE)
            logger.info(f"Basler camera {self.config.id} disconnected")
            return True
        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            return False

    def capture(self, sequence_id: str = "") -> Optional[CapturedImage]:
        if not self._is_connected:
            return None

        if self._camera is None:
            return self._mock_capture(sequence_id)

        try:
            self._update_status(CameraStatus.CAPTURING)
            grab_result = self._camera.GrabOne(1000)

            if grab_result.GrabSucceeded():
                image = grab_result.Array
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
                        "gain": self.config.gain
                    }
                )

                grab_result.Release()
                self._update_status(CameraStatus.ONLINE)
                return captured
            else:
                raise RuntimeError(f"Grab failed: {grab_result.ErrorCode}")

        except Exception as e:
            self._update_status(CameraStatus.ERROR, str(e))
            logger.error(f"Capture failed for Basler camera {self.config.id}: {e}")
            return None

    def _mock_capture(self, sequence_id: str = "") -> Optional[CapturedImage]:
        import cv2
        self._update_status(CameraStatus.CAPTURING)
        image = np.random.normal(128, 20, (self.config.height, self.config.width)).astype(np.uint8)
        cv2.putText(image, f"BASLER-{self.config.id}", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

        self._increment_trigger_count()
        captured = CapturedImage.create(
            camera_id=self.config.id,
            camera_position=self.config.position,
            raw_data=image,
            width=self.config.width,
            height=self.config.height,
            pixel_format=self.config.pixel_format,
            trigger_count=self._trigger_count,
            sequence_id=sequence_id
        )
        self._update_status(CameraStatus.ONLINE)
        return captured

    def set_exposure_time(self, exposure_time: int) -> bool:
        if not self._is_connected:
            return False
        try:
            if self._camera is not None:
                self._camera.ExposureTime.SetValue(exposure_time)
            self.config.exposure_time = exposure_time
            return True
        except Exception as e:
            logger.error(f"Failed to set exposure time: {e}")
            return False

    def set_gain(self, gain: float) -> bool:
        if not self._is_connected:
            return False
        try:
            if self._camera is not None:
                self._camera.Gain.SetValue(gain)
            self.config.gain = gain
            return True
        except Exception as e:
            logger.error(f"Failed to set gain: {e}")
            return False

    def software_trigger(self) -> bool:
        if not self._is_connected:
            return False
        try:
            if self._camera is not None:
                self._camera.TriggerSoftware()
            return True
        except Exception as e:
            logger.error(f"Software trigger failed: {e}")
            return False
