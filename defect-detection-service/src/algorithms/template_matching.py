from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import time
import os

from src.algorithms.base_algorithm import BaseDetectionAlgorithm
from src.utils.schemas import (
    Defect, BoundingBox, Point, DefectType, DefectSeverity,
    AlgorithmType, InferenceResult, ProductConfig, ROI
)
from src.utils.logger import Logger

logger = Logger().logger


class TemplateMatchingAlgorithm(BaseDetectionAlgorithm):
    def __init__(self):
        super().__init__("template_matching", AlgorithmType.TEMPLATE_MATCHING)
        self._templates: List[np.ndarray] = []
        self._template_paths: List[str] = []

    def initialize(self, params: Dict[str, Any]) -> bool:
        try:
            self._params = {
                "template_paths": params.get("template_paths", []),
                "threshold": params.get("threshold", 0.8),
                "match_method": params.get("match_method", "TM_CCOEFF_NORMED"),
                "scale_range": params.get("scale_range", [0.8, 1.2]),
                "scale_steps": params.get("scale_steps", 5),
                "defect_type": params.get("defect_type", DefectType.MISSING),
                "detect_missing": params.get("detect_missing", True),
                "detect_mismatch": params.get("detect_mismatch", True),
                "blur_kernel": params.get("blur_kernel", 3),
                "use_gray": params.get("use_gray", True),
                "min_match_area_pixels": params.get("min_match_area_pixels", 500)
            }

            template_paths = self._params["template_paths"]
            if not template_paths:
                error_msg = "❌ 模板匹配初始化失败：未配置任何模板路径 (template_paths is empty)"
                logger.error(error_msg)
                self._is_initialized = False
                return False

            self._load_templates()
            self._is_initialized = len(self._templates) > 0

            if self._is_initialized:
                logger.info(f"✅ 模板匹配初始化成功，加载 {len(self._templates)} 个模板")
                for i, path in enumerate(self._template_paths):
                    logger.info(f"   模板 [{i+1}]: {path}")
            else:
                error_msg = f"❌ 模板匹配初始化失败：{len(template_paths)} 个模板全部加载失败"
                error_msg += f"\n   尝试加载的模板路径："
                for path in template_paths:
                    exists = "✓" if os.path.exists(path) else "✗"
                    error_msg += f"\n     {exists} {path}"
                logger.error(error_msg)
                self._is_initialized = False
                return False

            return self._is_initialized
        except Exception as e:
            error_msg = f"❌ 模板匹配初始化异常：{str(e)}"
            logger.error(error_msg, exc_info=True)
            self._is_initialized = False
            return False

    def _load_templates(self):
        self._templates = []
        self._template_paths = []

        template_paths = self._params["template_paths"]
        for path in template_paths:
            if not os.path.exists(path):
                logger.error(f"✗ 模板文件不存在: {path}")
                continue

            try:
                template = cv2.imread(path)
                if template is not None:
                    if self._params["use_gray"] and len(template.shape) == 3:
                        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    self._templates.append(template)
                    self._template_paths.append(path)
                    logger.info(f"Loaded template: {path}, shape: {template.shape}")
                else:
                    logger.warning(f"Failed to read template: {path}")
            except Exception as e:
                logger.error(f"Error loading template {path}: {e}")

    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        start_time = time.time()
        defects: List[Defect] = []

        try:
            if not self._is_initialized:
                error_msg = "Template matching algorithm not initialized"
                logger.error(error_msg)
                return defects, InferenceResult(
                    success=False,
                    algorithm_type=self.algorithm_type,
                    error_message=error_msg
                )

            roi_image, offset = self._extract_roi(image, roi)
            search_image = roi_image.copy()

            if self._params["use_gray"] and len(search_image.shape) == 3:
                search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)

            blur_kernel = self._params["blur_kernel"]
            if blur_kernel > 0:
                search_image = cv2.GaussianBlur(
                    search_image, (blur_kernel, blur_kernel), 0
                )

            match_method = getattr(cv2, self._params["match_method"])
            threshold = self._params["threshold"]
            defect_type = self._params["defect_type"]

            all_matches: List[Tuple[float, Tuple[int, int, int, int], int]] = []

            for template_idx, template in enumerate(self._templates):
                th, tw = template.shape[:2]
                if th > search_image.shape[0] or tw > search_image.shape[1]:
                    continue

                scale_min, scale_max = self._params["scale_range"]
                scale_steps = self._params["scale_steps"]

                for scale in np.linspace(scale_min, scale_max, scale_steps):
                    scaled_template = cv2.resize(
                        template,
                        (int(tw * scale), int(th * scale)),
                        interpolation=cv2.INTER_AREA
                    )

                    sth, stw = scaled_template.shape[:2]
                    if sth > search_image.shape[0] or stw > search_image.shape[1]:
                        continue

                    result = cv2.matchTemplate(search_image, scaled_template, match_method)

                    if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        loc = np.where(result <= (1 - threshold))
                        confidence_base = 1.0
                    else:
                        loc = np.where(result >= threshold)
                        confidence_base = 0.0

                    for pt in zip(*loc[::-1]):
                        if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            confidence = 1.0 - result[pt[1], pt[0]]
                        else:
                            confidence = result[pt[1], pt[0]]

                        x, y = pt
                        w, h = stw, sth

                        area = w * h
                        if area < self._params["min_match_area_pixels"]:
                            continue

                        all_matches.append((confidence, (x, y, w, h), template_idx))

            all_matches.sort(key=lambda x: x[0], reverse=True)
            suppressed = self._non_max_suppression(all_matches)

            if self._params["detect_missing"] and len(suppressed) == 0:
                h, w = search_image.shape[:2]
                center_x = w // 2 + offset[0]
                center_y = h // 2 + offset[1]

                bbox = BoundingBox(
                    x1=float(offset[0]),
                    y1=float(offset[1]),
                    x2=float(offset[0] + w),
                    y2=float(offset[1] + h)
                )

                defect = Defect.create(
                    defect_type=defect_type,
                    severity=DefectSeverity.CRITICAL,
                    confidence=0.9,
                    bbox=bbox,
                    area_pixels=float(w * h),
                    area_mm2=float(w * h) * (product_config.pixel_to_mm_ratio ** 2),
                    description="Template not found - object missing"
                )
                defects.append(defect)

            elif self._params["detect_mismatch"]:
                for confidence, (x, y, w, h), template_idx in suppressed:
                    bbox = BoundingBox(
                        x1=float(x + offset[0]),
                        y1=float(y + offset[1]),
                        x2=float(x + w + offset[0]),
                        y2=float(y + h + offset[1])
                    )

                    defect = Defect.create(
                        defect_type=DefectType.DEFORMATION,
                        severity=DefectSeverity.MAJOR,
                        confidence=float(confidence),
                        bbox=bbox,
                        area_pixels=float(w * h),
                        area_mm2=float(w * h) * (product_config.pixel_to_mm_ratio ** 2),
                        description=f"Template mismatch with confidence: {confidence:.3f}"
                    )
                    defects.append(defect)

            inference_time = (time.time() - start_time) * 1000

            return defects, InferenceResult(
                success=True,
                inference_time_ms=inference_time,
                algorithm_type=self.algorithm_type,
                raw_output={
                    "match_count": len(all_matches),
                    "defect_count": len(defects),
                    "template_count": len(self._templates)
                }
            )

        except Exception as e:
            logger.error(f"Template matching failed: {e}", exc_info=True)
            inference_time = (time.time() - start_time) * 1000
            return defects, InferenceResult(
                success=False,
                inference_time_ms=inference_time,
                algorithm_type=self.algorithm_type,
                error_message=str(e)
            )

    def _non_max_suppression(self, matches: List[Tuple[float, Tuple[int, int, int, int], int]],
                             iou_threshold: float = 0.3) -> List[Tuple[float, Tuple[int, int, int, int], int]]:
        if len(matches) == 0:
            return []

        boxes = np.array([[x, y, x + w, y + h] for _, (x, y, w, h), _ in matches])
        confidences = np.array([c for c, _, _ in matches])

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

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

        return [matches[i] for i in pick]

    def add_template(self, template_path: str) -> bool:
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return False

        try:
            template = cv2.imread(template_path)
            if template is None:
                return False

            if self._params["use_gray"] and len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            self._templates.append(template)
            self._template_paths.append(template_path)
            self._is_initialized = True
            logger.info(f"Added template: {template_path}")
            return True
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
