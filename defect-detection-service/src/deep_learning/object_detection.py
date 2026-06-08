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


class ObjectDetectionAlgorithm(BaseDeepLearningAlgorithm):
    def __init__(self):
        super().__init__("object_detection", AlgorithmType.OBJECT_DETECTION)
        self._anchors: List[List[float]] = []

    def initialize(self, params: Dict[str, Any], model_path: str,
                   backend: str = "onnx_cpu", gpu_device_id: int = 0,
                   enable_tensorrt: bool = False) -> bool:
        if not super().initialize(params, model_path, backend, gpu_device_id, enable_tensorrt):
            return False

        self._anchors = params.get("anchors", [])
        self._strides = params.get("strides", [8, 16, 32])
        self._format = params.get("format", "yolo")

        return True

    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        start_time = time.time()
        defects: List[Defect] = []

        try:
            if not self._is_initialized or self._engine is None:
                error_msg = "Object detection algorithm not initialized"
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
            logger.error(f"Object detection failed: {e}", exc_info=True)
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

        if self._format == "yolo":
            detections = self._postprocess_yolo(outputs, input_shape)
        else:
            detections = self._postprocess_generic(outputs, input_shape)

        detections = self._nms(detections)

        conf_threshold = self._params.get("conf_threshold", 0.5)
        nms_threshold = self._params.get("nms_threshold", 0.45)

        for det in detections:
            x1, y1, x2, y2, conf, class_idx = det

            if conf < conf_threshold:
                continue

            bbox = self._scale_bbox((x1, y1, x2, y2), input_shape, original_shape, offset)
            area_pixels = bbox.area
            area_mm2 = area_pixels * (product_config.pixel_to_mm_ratio ** 2)

            class_idx = int(class_idx)
            defect_type = self._get_defect_type(class_idx)
            defect_config = product_config.get_defect_config(defect_type)

            if defect_config and not defect_config.enabled:
                continue

            if defect_config:
                if area_mm2 < defect_config.min_area_mm2 or area_mm2 > defect_config.max_area_mm2:
                    continue
                if conf < defect_config.min_confidence:
                    continue

            severity = self._get_defect_severity(defect_type, conf, product_config)

            class_name = self._class_names[class_idx] if class_idx < len(self._class_names) else str(class_idx)

            defect = Defect.create(
                defect_type=defect_type,
                severity=severity,
                confidence=float(conf),
                bbox=bbox,
                area_pixels=float(area_pixels),
                area_mm2=float(area_mm2),
                description=f"{class_name}: {conf:.3f}, area: {area_mm2:.4f} mm²",
                metadata={"class_idx": class_idx, "class_name": class_name}
            )

            defects.append(defect)

        return defects

    def _postprocess_yolo(self, outputs: Dict[str, np.ndarray], input_shape: Tuple[int, int]) -> List[List[float]]:
        detections: List[List[float]] = []

        output_name = self._params.get("output_name", None)
        if output_name is None and outputs:
            output_name = list(outputs.keys())[0]

        if output_name not in outputs:
            return detections

        pred = outputs[output_name]
        batch_size = pred.shape[0]

        for batch in range(batch_size):
            batch_pred = pred[batch]

            if len(batch_pred.shape) == 2:
                num_predictions = batch_pred.shape[0]
                for i in range(num_predictions):
                    x, y, w, h, obj_conf = batch_pred[i, :5]
                    class_preds = batch_pred[i, 5:]

                    if len(class_preds) > 0:
                        class_idx = np.argmax(class_preds)
                        class_conf = class_preds[class_idx]
                        conf = obj_conf * class_conf
                    else:
                        class_idx = 1
                        conf = obj_conf

                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2

                    x1 *= input_shape[1]
                    y1 *= input_shape[0]
                    x2 *= input_shape[1]
                    y2 *= input_shape[0]

                    detections.append([x1, y1, x2, y2, float(conf), float(class_idx)])

            elif len(batch_pred.shape) == 3:
                for i in range(batch_pred.shape[0]):
                    for j in range(batch_pred.shape[1]):
                        x, y, w, h, obj_conf = batch_pred[i, j, :5]
                        class_preds = batch_pred[i, j, 5:]

                        if obj_conf < 0.001:
                            continue

                        if len(class_preds) > 0:
                            class_idx = np.argmax(class_preds)
                            class_conf = class_preds[class_idx]
                            conf = obj_conf * class_conf
                        else:
                            class_idx = 1
                            conf = obj_conf

                        if conf < 0.001:
                            continue

                        stride = self._strides[min(i, len(self._strides) - 1)]
                        grid_x = j % (input_shape[1] // stride)
                        grid_y = j // (input_shape[1] // stride)

                        x = (x * 2 - 0.5 + grid_x) * stride
                        y = (y * 2 - 0.5 + grid_y) * stride
                        w = (w * 2) ** 2 * (self._anchors[i * 2] if len(self._anchors) > i * 2 else 32)
                        h = (h * 2) ** 2 * (self._anchors[i * 2 + 1] if len(self._anchors) > i * 2 + 1 else 32)

                        x1 = x - w / 2
                        y1 = y - h / 2
                        x2 = x + w / 2
                        y2 = y + h / 2

                        detections.append([x1, y1, x2, y2, float(conf), float(class_idx)])

        return detections

    def _postprocess_generic(self, outputs: Dict[str, np.ndarray], input_shape: Tuple[int, int]) -> List[List[float]]:
        detections: List[List[float]] = []

        boxes_output = self._params.get("boxes_output", "boxes")
        scores_output = self._params.get("scores_output", "scores")
        labels_output = self._params.get("labels_output", "labels")

        if boxes_output in outputs:
            boxes = outputs[boxes_output][0]
            scores = outputs[scores_output][0] if scores_output in outputs else np.ones(len(boxes))
            labels = outputs[labels_output][0] if labels_output in outputs else np.ones(len(boxes))

            for i in range(len(boxes)):
                if len(boxes[i]) == 4:
                    y1, x1, y2, x2 = boxes[i]
                else:
                    x1, y1, x2, y2 = boxes[i]

                conf = float(scores[i]) if i < len(scores) else 1.0
                class_idx = int(labels[i]) if i < len(labels) else 1

                detections.append([
                    float(x1) * input_shape[1],
                    float(y1) * input_shape[0],
                    float(x2) * input_shape[1],
                    float(y2) * input_shape[0],
                    conf,
                    float(class_idx)
                ])

        return detections

    def _nms(self, detections: List[List[float]], iou_threshold: float = 0.45) -> List[List[float]]:
        if len(detections) == 0:
            return []

        detections = np.array(detections)
        pick = []

        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        confidences = detections[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))

        return [detections[i].tolist() for i in pick]
