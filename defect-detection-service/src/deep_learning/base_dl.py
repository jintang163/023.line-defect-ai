from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import time
import os

from src.inference.onnx_engine import ONNXInferenceEngine
from src.utils.schemas import (
    Defect, BoundingBox, Point, DefectType, DefectSeverity,
    AlgorithmType, InferenceResult, ProductConfig, ROI
)
from src.utils.logger import Logger

logger = Logger().logger


class BaseDeepLearningAlgorithm(ABC):
    def __init__(self, name: str, algorithm_type: AlgorithmType):
        self.name = name
        self.algorithm_type = algorithm_type
        self._engine: Optional[ONNXInferenceEngine] = None
        self._params: Dict[str, Any] = {}
        self._model_path: str = ""
        self._is_initialized = False
        self._class_names: List[str] = []
        self._defect_type_map: Dict[int, DefectType] = {}

    def initialize(self, params: Dict[str, Any], model_path: str,
                   backend: str = "onnx_cpu", gpu_device_id: int = 0,
                   enable_tensorrt: bool = False) -> bool:
        try:
            self._params = params
            self._model_path = model_path

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            from src.utils.schemas import InferenceBackend
            backend_enum = InferenceBackend(backend)

            self._engine = ONNXInferenceEngine(
                model_path=model_path,
                backend=backend_enum,
                gpu_device_id=gpu_device_id,
                enable_tensorrt=enable_tensorrt,
                enable_dynamic_batch=params.get("enable_dynamic_batch", False),
                max_batch_size=params.get("max_batch_size", 16)
            )

            if not self._engine.initialize():
                logger.error("Failed to initialize inference engine")
                return False

            self._setup_class_mappings()
            self._is_initialized = True
            logger.info(f"{self.name} initialized with model: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}", exc_info=True)
            self._is_initialized = False
            return False

    @abstractmethod
    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        pass

    @abstractmethod
    def _postprocess(self, outputs: Dict[str, np.ndarray], original_shape: Tuple[int, int],
                     input_shape: Tuple[int, int], offset: Tuple[int, int],
                     product_config: ProductConfig) -> List[Defect]:
        pass

    def _setup_class_mappings(self):
        self._class_names = self._params.get("class_names", [])

        defect_type_map = self._params.get("defect_type_map", {})
        self._defect_type_map = {}
        for class_idx, defect_type_str in defect_type_map.items():
            try:
                self._defect_type_map[int(class_idx)] = DefectType(defect_type_str)
            except (ValueError, KeyError):
                logger.warning(f"Invalid defect type mapping: {class_idx} -> {defect_type_str}")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        target_size = self._params.get("input_size", (224, 224))
        mean = self._params.get("mean", [0.485, 0.456, 0.406])
        std = self._params.get("std", [0.229, 0.224, 0.225])
        normalize = self._params.get("normalize", True)

        original_shape = image.shape[:2]

        processed = self._engine.preprocess_image(
            image,
            target_size=target_size,
            mean=mean,
            std=std,
            normalize=normalize,
            to_chw=True
        )

        return np.expand_dims(processed, axis=0), original_shape

    def _get_defect_type(self, class_idx: int) -> DefectType:
        if class_idx in self._defect_type_map:
            return self._defect_type_map[class_idx]

        if class_idx < len(self._class_names):
            class_name = self._class_names[class_idx].lower()
            for defect_type in DefectType:
                if defect_type.value in class_name:
                    return defect_type

        return DefectType.UNKNOWN

    def _get_defect_severity(self, defect_type: DefectType, confidence: float,
                             product_config: ProductConfig) -> DefectSeverity:
        defect_config = product_config.get_defect_config(defect_type)
        if defect_config:
            return defect_config.severity

        if confidence >= 0.9:
            return DefectSeverity.CRITICAL
        elif confidence >= 0.7:
            return DefectSeverity.MAJOR
        elif confidence >= 0.5:
            return DefectSeverity.MINOR
        else:
            return DefectSeverity.WARNING

    def _extract_roi(self, image: np.ndarray, roi: Optional[ROI]) -> Tuple[np.ndarray, Tuple[int, int]]:
        if roi is None or not roi.enabled:
            return image, (0, 0)

        x, y, w, h = roi.to_tuple()
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        if w <= 0 or h <= 0:
            return image, (0, 0)

        return image[y:y + h, x:x + w], (x, y)

    def _scale_bbox(self, bbox: Tuple[float, float, float, float],
                    input_shape: Tuple[int, int], original_shape: Tuple[int, int],
                    offset: Tuple[int, int]) -> BoundingBox:
        input_h, input_w = input_shape
        orig_h, orig_w = original_shape
        offset_x, offset_y = offset

        x1, y1, x2, y2 = bbox

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        return BoundingBox(
            x1=float(x1 * scale_x + offset_x),
            y1=float(y1 * scale_y + offset_y),
            x2=float(x2 * scale_x + offset_x),
            y2=float(y2 * scale_y + offset_y)
        )

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def destroy(self):
        if self._engine:
            self._engine.destroy()
        self._is_initialized = False
