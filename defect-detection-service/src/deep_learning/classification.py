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


class ClassificationAlgorithm(BaseDeepLearningAlgorithm):
    def __init__(self):
        super().__init__("classification", AlgorithmType.CLASSIFICATION)

    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        start_time = time.time()
        defects: List[Defect] = []

        try:
            if not self._is_initialized or self._engine is None:
                error_msg = "Classification algorithm not initialized"
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
            logger.error(f"Classification detection failed: {e}", exc_info=True)
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

        predictions = outputs[output_name]
        predictions = np.squeeze(predictions)

        if len(predictions.shape) == 0:
            predictions = np.array([predictions])

        if len(predictions.shape) == 1 and predictions.shape[0] == 1:
            prob = float(predictions[0])
            if prob > 0.5:
                class_idx = 1
                confidence = prob
            else:
                class_idx = 0
                confidence = 1 - prob
        else:
            class_idx = int(np.argmax(predictions))
            confidence = float(predictions[class_idx])

        threshold = self._params.get("threshold", 0.5)
        if confidence < threshold:
            return defects

        ok_class_idx = self._params.get("ok_class_idx", 0)
        if class_idx == ok_class_idx:
            return defects

        defect_type = self._get_defect_type(class_idx)
        defect_config = product_config.get_defect_config(defect_type)

        if defect_config and not defect_config.enabled:
            return defects

        if defect_config and confidence < defect_config.min_confidence:
            return defects

        severity = self._get_defect_severity(defect_type, confidence, product_config)

        orig_h, orig_w = original_shape
        offset_x, offset_y = offset

        bbox = BoundingBox(
            x1=float(offset_x),
            y1=float(offset_y),
            x2=float(orig_w + offset_x),
            y2=float(orig_h + offset_y)
        )

        class_name = self._class_names[class_idx] if class_idx < len(self._class_names) else str(class_idx)

        defect = Defect.create(
            defect_type=defect_type,
            severity=severity,
            confidence=confidence,
            bbox=bbox,
            area_pixels=float(orig_w * orig_h),
            area_mm2=float(orig_w * orig_h) * (product_config.pixel_to_mm_ratio ** 2),
            description=f"Classification: {class_name} ({confidence:.3f})",
            metadata={"class_idx": class_idx, "class_name": class_name}
        )

        defects.append(defect)
        return defects

    def get_class_probabilities(self, image: np.ndarray, roi: Optional[ROI] = None) -> Dict[str, float]:
        if not self._is_initialized or self._engine is None:
            return {}

        try:
            roi_image, _ = self._extract_roi(image, roi)
            input_tensor, _ = self._preprocess(roi_image)

            input_name = self._engine.input_names[0]
            outputs, _ = self._engine.infer({input_name: input_tensor})

            output_name = self._params.get("output_name", None)
            if output_name is None and outputs:
                output_name = list(outputs.keys())[0]

            if output_name not in outputs:
                return {}

            predictions = np.squeeze(outputs[output_name])
            if len(predictions.shape) == 0:
                predictions = np.array([1 - predictions, predictions])

            probs = {}
            for i, prob in enumerate(predictions):
                if i < len(self._class_names):
                    probs[self._class_names[i]] = float(prob)
                else:
                    probs[f"class_{i}"] = float(prob)

            return probs

        except Exception as e:
            logger.error(f"Failed to get class probabilities: {e}")
            return {}
