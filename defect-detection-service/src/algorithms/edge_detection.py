from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import time

from src.algorithms.base_algorithm import BaseDetectionAlgorithm
from src.utils.schemas import (
    Defect, BoundingBox, Point, DefectType, DefectSeverity,
    AlgorithmType, InferenceResult, ProductConfig, ROI
)
from src.utils.logger import Logger

logger = Logger().logger


class EdgeDetectionAlgorithm(BaseDetectionAlgorithm):
    def __init__(self):
        super().__init__("edge_detection", AlgorithmType.EDGE_DETECTION)

    def initialize(self, params: Dict[str, Any]) -> bool:
        try:
            self._params = {
                "low_threshold": params.get("low_threshold", 50),
                "high_threshold": params.get("high_threshold", 150),
                "aperture_size": params.get("aperture_size", 3),
                "l2_gradient": params.get("l2_gradient", False),
                "blur_kernel": params.get("blur_kernel", 5),
                "min_contour_area_pixels": params.get("min_contour_area_pixels", 100),
                "max_contour_area_pixels": params.get("max_contour_area_pixels", 100000),
                "defect_type": params.get("defect_type", DefectType.SCRATCH),
                "morphological_kernel": params.get("morphological_kernel", 3)
            }
            self._is_initialized = True
            logger.info(f"Edge detection algorithm initialized with params: {self._params}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize edge detection: {e}")
            self._is_initialized = False
            return False

    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        start_time = time.time()
        defects: List[Defect] = []

        try:
            if not self._is_initialized:
                error_msg = "Edge detection algorithm not initialized"
                logger.error(error_msg)
                return defects, InferenceResult(
                    success=False,
                    algorithm_type=self.algorithm_type,
                    error_message=error_msg
                )

            roi_image, offset = self._extract_roi(image, roi)

            if len(roi_image.shape) == 3:
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_image

            blur_kernel = self._params["blur_kernel"]
            if blur_kernel > 0:
                gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

            edges = cv2.Canny(
                gray,
                self._params["low_threshold"],
                self._params["high_threshold"],
                apertureSize=self._params["aperture_size"],
                L2gradient=self._params["l2_gradient"]
            )

            morph_kernel = self._params["morphological_kernel"]
            if morph_kernel > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            min_area = self._params["min_contour_area_pixels"]
            max_area = self._params["max_contour_area_pixels"]
            defect_type = self._params["defect_type"]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                confidence = min(1.0, area / min_area * 0.5 + 0.5)

                defect = self._create_defect(
                    contour, offset, defect_type, confidence, product_config
                )
                if defect:
                    defects.append(defect)

            inference_time = (time.time() - start_time) * 1000

            return defects, InferenceResult(
                success=True,
                inference_time_ms=inference_time,
                algorithm_type=self.algorithm_type,
                raw_output={"edge_count": len(contours), "defect_count": len(defects)}
            )

        except Exception as e:
            logger.error(f"Edge detection failed: {e}", exc_info=True)
            inference_time = (time.time() - start_time) * 1000
            return defects, InferenceResult(
                success=False,
                inference_time_ms=inference_time,
                algorithm_type=self.algorithm_type,
                error_message=str(e)
            )
