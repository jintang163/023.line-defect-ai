import threading
import time
import uuid
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from src.capture.base_camera import BaseCamera
from src.capture.mock_camera import MockCamera
from src.capture.basler_camera import BaslerCamera
from src.capture.hik_camera import HikvisionCamera
from src.config.settings import ConfigManager
from src.utils.schemas import (
    CameraConfig, CameraType, CameraStatusInfo, CapturedImage, ImageMessage
)
from src.utils.logger import Logger

logger = Logger().logger


class CameraManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._cameras: Dict[str, BaseCamera] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._trigger_callbacks: List[Callable[[ImageMessage], None]] = []
        self._is_running = False
        self._trigger_count = 0

    def initialize(self) -> bool:
        camera_configs = self.config_manager.get_camera_configs()
        success_count = 0

        for cam_config in camera_configs:
            camera = self._create_camera(cam_config)
            if camera.connect():
                self._cameras[cam_config.id] = camera
                success_count += 1
            else:
                logger.error(f"Failed to initialize camera {cam_config.id}")

        logger.info(f"Initialized {success_count}/{len(camera_configs)} cameras")
        return success_count > 0

    def _create_camera(self, config: CameraConfig) -> BaseCamera:
        if config.type == CameraType.BASLER:
            return BaslerCamera(config)
        elif config.type == CameraType.HIKVISION:
            return HikvisionCamera(config)
        else:
            return MockCamera(config)

    def add_trigger_callback(self, callback: Callable[[ImageMessage], None]):
        self._trigger_callbacks.append(callback)

    def trigger_sync_capture(self, product_id: Optional[str] = None) -> Optional[ImageMessage]:
        if not self._cameras:
            logger.warning("No cameras available for capture")
            return None

        sequence_id = str(uuid.uuid4())
        self._trigger_count += 1

        logger.info(f"Sync trigger #{self._trigger_count}, sequence: {sequence_id}")

        futures: Dict[str, Future] = {}
        with self._lock:
            for cam_id, camera in self._cameras.items():
                future = self._executor.submit(
                    self._capture_single, camera, sequence_id
                )
                futures[cam_id] = future

        captured_images: List[CapturedImage] = []
        for cam_id, future in futures.items():
            try:
                result = future.result(timeout=5.0)
                if result:
                    captured_images.append(result)
                else:
                    logger.warning(f"Camera {cam_id} returned no image")
            except Exception as e:
                logger.error(f"Capture from camera {cam_id} failed: {e}")

        if not captured_images:
            logger.error("No images captured from any camera")
            return None

        message = ImageMessage(
            sequence_id=sequence_id,
            timestamp=time.time(),
            images=captured_images,
            product_id=product_id
        )

        for callback in self._trigger_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Trigger callback failed: {e}")

        logger.info(f"Captured {len(captured_images)} images for sequence {sequence_id}")
        return message

    def _capture_single(self, camera: BaseCamera, sequence_id: str) -> Optional[CapturedImage]:
        return camera.capture(sequence_id)

    def get_camera_statuses(self) -> List[CameraStatusInfo]:
        statuses = []
        with self._lock:
            for camera in self._cameras.values():
                statuses.append(camera.get_status())
        return statuses

    def get_camera(self, camera_id: str) -> Optional[BaseCamera]:
        return self._cameras.get(camera_id)

    def set_camera_exposure(self, camera_id: str, exposure_time: int) -> bool:
        camera = self._cameras.get(camera_id)
        if camera:
            return camera.set_exposure_time(exposure_time)
        return False

    def set_camera_gain(self, camera_id: str, gain: float) -> bool:
        camera = self._cameras.get(camera_id)
        if camera:
            return camera.set_gain(gain)
        return False

    def set_all_exposure(self, exposure_time: int) -> Dict[str, bool]:
        results = {}
        with self._lock:
            for cam_id, camera in self._cameras.items():
                results[cam_id] = camera.set_exposure_time(exposure_time)
        return results

    def set_all_gain(self, gain: float) -> Dict[str, bool]:
        results = {}
        with self._lock:
            for cam_id, camera in self._cameras.items():
                results[cam_id] = camera.set_gain(gain)
        return results

    def software_trigger_all(self) -> bool:
        success = True
        with self._lock:
            for camera in self._cameras.values():
                if not camera.software_trigger():
                    success = False
        return success

    def disconnect_all(self):
        logger.info("Disconnecting all cameras...")
        with self._lock:
            for camera in self._cameras.values():
                try:
                    camera.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting camera: {e}")
            self._cameras.clear()
        self._executor.shutdown(wait=True)
        logger.info("All cameras disconnected")

    def reconnect_camera(self, camera_id: str) -> bool:
        camera = self._cameras.get(camera_id)
        if not camera:
            logger.warning(f"Camera {camera_id} not found for reconnection")
            return False

        try:
            camera.disconnect()
            time.sleep(0.5)
            return camera.connect()
        except Exception as e:
            logger.error(f"Error reconnecting camera {camera_id}: {e}")
            return False

    def health_check(self) -> Dict[str, bool]:
        results = {}
        with self._lock:
            for cam_id, camera in self._cameras.items():
                results[cam_id] = camera.is_connected()
        return results

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def cameras(self) -> Dict[str, BaseCamera]:
        return self._cameras.copy()
