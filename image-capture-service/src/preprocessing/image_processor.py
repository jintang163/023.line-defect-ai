import cv2
import numpy as np
from typing import Optional, Dict, Any
from src.config.settings import ConfigManager
from src.utils.schemas import CapturedImage, ROI, DistortionParams
from src.utils.logger import Logger

logger = Logger().logger


class ImagePreprocessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._load_config()
        self._map1 = None
        self._map2 = None
        self._init_undistort_maps()

    def _load_config(self):
        cfg = self.config_manager.get_preprocessing_config()
        self.enable_bayer = cfg.get("enable_bayer_conversion", True)
        self.enable_undistort = cfg.get("enable_undistort", True)
        self.enable_roi = cfg.get("enable_roi_crop", True)
        self.enable_resize = cfg.get("enable_resize", True)

        self.roi = self.config_manager.get_roi()
        self.distortion_params = self.config_manager.get_distortion_params()

        resize_cfg = cfg.get("resize", {})
        self.target_width = resize_cfg.get("width", 960)
        self.target_height = resize_cfg.get("height", 600)
        self.interpolation = self._get_interpolation(resize_cfg.get("interpolation", "linear"))

    def _init_undistort_maps(self):
        if not self.enable_undistort:
            return

        try:
            h, w = 1200, 1920
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.distortion_params.camera_matrix,
                self.distortion_params.dist_coeffs,
                (w, h), 0, (w, h)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.distortion_params.camera_matrix,
                self.distortion_params.dist_coeffs,
                None, new_camera_matrix, (w, h), cv2.CV_32FC1
            )
            self._undistort_roi = roi
            logger.info("Undistortion maps initialized")
        except Exception as e:
            logger.error(f"Failed to initialize undistort maps: {e}")
            self.enable_undistort = False

    def _get_interpolation(self, method: str) -> int:
        methods = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        return methods.get(method.lower(), cv2.INTER_LINEAR)

    def process(self, image: CapturedImage) -> Optional[np.ndarray]:
        if image.raw_data is None:
            logger.warning("Raw image data is None")
            return None

        processed = image.raw_data.copy()
        processing_steps = []

        try:
            if self.enable_bayer and self._is_bayer_format(image.pixel_format):
                processed = self._bayer_to_rgb(processed, image.pixel_format)
                processing_steps.append("bayer_conversion")

            if self.enable_undistort:
                processed = self._undistort(processed)
                processing_steps.append("undistort")

            if self.enable_roi and self.roi.width > 0 and self.roi.height > 0:
                processed = self._crop_roi(processed, self.roi)
                processing_steps.append("roi_crop")

            if self.enable_resize:
                processed = self._resize(processed, self.target_width, self.target_height)
                processing_steps.append("resize")

            image.processed_data = processed
            image.metadata["preprocessing_steps"] = processing_steps
            image.metadata["processed_width"] = processed.shape[1]
            image.metadata["processed_height"] = processed.shape[0]

            logger.debug(f"Image {image.image_id} processed with steps: {processing_steps}")
            return processed

        except Exception as e:
            logger.error(f"Preprocessing failed for image {image.image_id}: {e}")
            return None

    def process_message(self, message) -> None:
        for img in message.images:
            self.process(img)

    def _is_bayer_format(self, pixel_format: str) -> bool:
        bayer_formats = ["BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8",
                         "BayerRG12", "BayerBG12", "BayerGR12", "BayerGB12"]
        return pixel_format in bayer_formats

    def _bayer_to_rgb(self, image: np.ndarray, pixel_format: str) -> np.ndarray:
        if len(image.shape) == 3:
            return image

        pattern_map = {
            "BayerRG8": cv2.COLOR_BayerRG2RGB,
            "BayerBG8": cv2.COLOR_BayerBG2RGB,
            "BayerGR8": cv2.COLOR_BayerGR2RGB,
            "BayerGB8": cv2.COLOR_BayerGB2RGB,
            "BayerRG12": cv2.COLOR_BayerRG2RGB,
            "BayerBG12": cv2.COLOR_BayerBG2RGB,
            "BayerGR12": cv2.COLOR_BayerGR2RGB,
            "BayerGB12": cv2.COLOR_BayerGB2RGB,
        }
        code = pattern_map.get(pixel_format, cv2.COLOR_BayerRG2RGB)
        return cv2.cvtColor(image, code)

    def _undistort(self, image: np.ndarray) -> np.ndarray:
        if self._map1 is None or self._map2 is None:
            return image

        h, w = image.shape[:2]
        map_h, map_w = self._map1.shape[:2]

        if h != map_h or w != map_w:
            self._reinit_undistort_maps(w, h)

        undistorted = cv2.remap(image, self._map1, self._map2, self.interpolation)
        return undistorted

    def _reinit_undistort_maps(self, w: int, h: int):
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.distortion_params.camera_matrix,
            self.distortion_params.dist_coeffs,
            (w, h), 0, (w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.distortion_params.camera_matrix,
            self.distortion_params.dist_coeffs,
            None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
        self._undistort_roi = roi

    def _crop_roi(self, image: np.ndarray, roi: ROI) -> np.ndarray:
        h, w = image.shape[:2]
        x, y, rw, rh = roi.to_tuple()

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

        return image[y:y + rh, x:x + rw]

    def _resize(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = image.shape[:2]
        if w == target_w and h == target_h:
            return image
        return cv2.resize(image, (target_w, target_h), interpolation=self.interpolation)

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        if "roi" in config_updates:
            roi_cfg = config_updates["roi"]
            self.roi = ROI(**roi_cfg)
            self.enable_roi = True
            logger.info(f"ROI updated: {self.roi}")

        if "resize" in config_updates:
            resize_cfg = config_updates["resize"]
            self.target_width = resize_cfg.get("width", self.target_width)
            self.target_height = resize_cfg.get("height", self.target_height)
            self.interpolation = self._get_interpolation(resize_cfg.get("interpolation", "linear"))
            self.enable_resize = True
            logger.info(f"Resize config updated: {self.target_width}x{self.target_height}")

        if "enable_undistort" in config_updates:
            self.enable_undistort = config_updates["enable_undistort"]
            logger.info(f"Undistort enabled: {self.enable_undistort}")

    def reload_config(self):
        self._load_config()
        self._init_undistort_maps()
        logger.info("Preprocessor config reloaded")
