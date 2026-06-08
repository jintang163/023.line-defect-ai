from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import time

from src.deep_learning.base_dl import BaseDeepLearningAlgorithm
from src.utils.schemas import (
    Defect, BoundingBox, Point, DefectType, DefectSeverity,
    AlgorithmType, InferenceResult, ProductConfig, ROI
)
from src.utils.logger import Logger

logger = Logger().logger


class SegmentationAlgorithm(BaseDeepLearningAlgorithm):
    def __init__(self):
        super().__init__("segmentation", AlgorithmType.SEGMENTATION)

    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        start_time = time.time()
        defects: List[Defect] = []

        try:
            if not self._is_initialized or self._engine is None:
                error_msg = "Segmentation algorithm not initialized"
                logger.error(error_msg)
                return defects, InferenceResult(
                    success=False,
                    algorithm_type=self.algorithm_type,
                    error_message=error_msg
                )

            roi_image, offset = self._extract_roi(image, roi)
            input_tensor, original_shape = self._preprocess(roi_image)

            input_name = self._engine.input_names[0]
            outputs, inference_time = self._engine.infer({input_name: input_tensor})

            defects = self._postprocess(
                outputs, original_shape,
                input_tensor.shape[2:],
                offset, product_config
            )

            total_time = (time.time() - start_time) * 1000

            return defects, InferenceResult(
                success=True,
                inference_time_ms=total_time,
                algorithm_type=self.algorithm_type,
                raw_output={
                    "output_shape": {k: v.shape for k, v in outputs.items()},
                    "defect_count": len(defects)
                }
            )

        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            total_time = (time.time() - start_time) * 1000
            return defects, InferenceResult(
                success=False,
                inference_time_ms=total_time,
                algorithm_type=self.algorithm_type,
                error_message=str(e)
            )

    def _postprocess(self, outputs: Dict[str, np.ndarray], original_shape: Tuple[int, int],
                     input_shape: Tuple[int, int], offset: Tuple[int, int],
                     product_config: ProductConfig) -> List[Defect]:
        defects: List[Defect] = []

        if not outputs:
            return defects

        output_name = self._params.get("output_name", None)
        if output_name is None and outputs:
            output_name = list(outputs.keys())[0]

        if output_name not in outputs:
            logger.error(f"Output {output_name} not found in model outputs")
            return defects

        mask = outputs[output_name]
        mask = np.squeeze(mask)

        if len(mask.shape) == 3:
            num_classes = mask.shape[0]
            if num_classes > 1:
                mask = np.argmax(mask, axis=0)
            else:
                mask = (mask[0] > self._params.get("threshold", 0.5)).astype(np.uint8)
        elif len(mask.shape) == 2:
            mask = (mask > self._params.get("threshold", 0.5)).astype(np.uint8)

        if mask.shape[0] != original_shape[0] or mask.shape[1] != original_shape[1]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        background_class = self._params.get("background_class", 0)
        min_area = self._params.get("min_area_pixels", 100)

        offset_x, offset_y = offset

        unique_classes = np.unique(mask)
        for class_idx in unique_classes:
            if class_idx == background_class:
                continue

            class_mask = (mask == class_idx).astype(np.uint8)

            morph_kernel = self._params.get("morph_kernel", 3)
            if morph_kernel > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(
                class_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                if area_pixels < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                bbox = BoundingBox(
                    x1=float(x + offset_x),
                    y1=float(y + offset_y),
                    x2=float(x + w + offset_x),
                    y2=float(y + h + offset_y)
                )

                area_mm2 = area_pixels * (product_config.pixel_to_mm_ratio ** 2)

                class_idx_int = int(class_idx)
                defect_type = self._get_defect_type(class_idx_int)
                defect_config = product_config.get_defect_config(defect_type)

                if defect_config and not defect_config.enabled:
                    continue

                if defect_config:
                    if area_mm2 < defect_config.min_area_mm2 or area_mm2 > defect_config.max_area_mm2:
                        continue

                mask_region = class_mask[y:y + h, x:x + w]
                mean_conf = float(np.mean(mask_region))
                confidence = min(1.0, max(0.3, mean_conf))

                if defect_config and confidence < defect_config.min_confidence:
                    continue

                severity = self._get_defect_severity(defect_type, confidence, product_config)

                points = [Point(float(p[0][0] + offset_x), float(p[0][1] + offset_y)) for p in contour]

                class_name = self._class_names[class_idx_int] if class_idx_int < len(self._class_names) else str(class_idx_int)

                defect = Defect.create(
                    defect_type=defect_type,
                    severity=severity,
                    confidence=confidence,
                    bbox=bbox,
                    contour=points,
                    area_pixels=float(area_pixels),
                    area_mm2=float(area_mm2),
                    description=f"{class_name}: area {area_mm2:.4f} mm², conf {confidence:.3f}",
                    metadata={"class_idx": class_idx_int, "class_name": class_name}
                )

                defects.append(defect)

        return defects

    def get_segmentation_mask(self, image: np.ndarray, roi: Optional[ROI] = None) -> Optional[np.ndarray]:
        if not self._is_initialized or self._engine is None:
            return None

        try:
            roi_image, offset = self._extract_roi(image, roi)
            input_tensor, original_shape = self._preprocess(roi_image)

            input_name = self._engine.input_names[0]
            outputs, _ = self._engine.infer({input_name: input_tensor})

            output_name = self._params.get("output_name", None)
            if output_name is None and outputs:
                output_name = list(outputs.keys())[0]

            if output_name not in outputs:
                return None

            mask = outputs[output_name]
            mask = np.squeeze(mask)

            if len(mask.shape) == 3:
                num_classes = mask.shape[0]
                if num_classes > 1:
                    mask = np.argmax(mask, axis=0)
                else:
                    mask = (mask[0] > self._params.get("threshold", 0.5)).astype(np.uint8)

            if mask.shape[0] != original_shape[0] or mask.shape[1] != original_shape[1]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_shape[1], original_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            return mask

        except Exception as e:
            logger.error(f"Failed to get segmentation mask: {e}")
            return None
