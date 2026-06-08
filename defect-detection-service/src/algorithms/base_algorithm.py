from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time

from src.utils.schemas import (
    Defect, BoundingBox, Point, DefectType, DefectSeverity,
    AlgorithmType, InferenceResult, ProductConfig, ROI, ImageData
)
from src.utils.logger import Logger

logger = Logger().logger


class BaseDetectionAlgorithm(ABC):
    def __init__(self, name: str, algorithm_type: AlgorithmType):
        self.name = name
        self.algorithm_type = algorithm_type
        self._params: Dict[str, Any] = {}
        self._is_initialized = False

    @abstractmethod
    def initialize(self, params: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        pass

    def set_param(self, key: str, value: Any):
        self._params[key] = value

    def get_param(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

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

    def _create_defect(self, contour: np.ndarray, offset: Tuple[int, int],
                       defect_type: DefectType, confidence: float,
                       product_config: ProductConfig) -> Optional[Defect]:
        if len(contour) < 3:
            return None

        offset_x, offset_y = offset

        x, y, w, h = cv2.boundingRect(contour)
        bbox = BoundingBox(
            x1=float(x + offset_x),
            y1=float(y + offset_y),
            x2=float(x + w + offset_x),
            y2=float(y + h + offset_y)
        )

        area_pixels = cv2.contourArea(contour)
        area_mm2 = area_pixels * (product_config.pixel_to_mm_ratio ** 2)

        defect_config = product_config.get_defect_config(defect_type)
        if defect_config and not defect_config.enabled:
            return None

        if defect_config:
            if area_mm2 < defect_config.min_area_mm2 or area_mm2 > defect_config.max_area_mm2:
                return None
            if confidence < defect_config.min_confidence:
                return None
            severity = defect_config.severity
        else:
            severity = DefectSeverity.MAJOR

        points = [Point(float(p[0][0] + offset_x), float(p[0][1] + offset_y)) for p in contour]

        return Defect.create(
            defect_type=defect_type,
            severity=severity,
            confidence=confidence,
            bbox=bbox,
            contour=points,
            area_pixels=area_pixels,
            area_mm2=area_mm2,
            description=f"{defect_type.value} detected, area: {area_mm2:.4f} mm²"
        )


import cv2
