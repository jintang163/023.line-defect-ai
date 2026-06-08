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


class GrayDiffAlgorithm(BaseDetectionAlgorithm):
    def __init__(self):
        super().__init__("gray_diff", AlgorithmType.GRAY_DIFF)
        self._reference_image: Optional[np.ndarray] = None
        self._reference_image_path: str = ""

    def initialize(self, params: Dict[str, Any]) -> bool:
        try:
            self._params = {
                "reference_image_path": params.get("reference_image_path", ""),
                "diff_threshold": params.get("diff_threshold", 30),
                "min_area_pixels": params.get("min_area_pixels", 50),
                "max_area_pixels": params.get("max_area_pixels", 50000),
                "defect_type": params.get("defect_type", DefectType.DIRT),
                "blur_kernel": params.get("blur_kernel", 5),
                "morph_kernel": params.get("morph_kernel", 3),
                "use_abs_diff": params.get("use_abs_diff", True),
                "normalize": params.get("normalize", True),
                "adaptive_threshold": params.get("adaptive_threshold", False),
                "adaptive_block_size": params.get("adaptive_block_size", 11),
                "adaptive_C": params.get("adaptive_C", 2),
                "register_images": params.get("register_images", True)
            }

            ref_path = self._params["reference_image_path"]

            if not ref_path:
                error_msg = "❌ 灰度差分初始化失败：未配置参考图像路径 (reference_image_path is empty)"
                logger.error(error_msg)
                self._is_initialized = False
                return False

            if not os.path.exists(ref_path):
                error_msg = f"❌ 灰度差分初始化失败：参考图像不存在\n   路径: {ref_path}"
                error_msg += f"\n   当前工作目录: {os.getcwd()}"
                error_msg += f"\n   绝对路径: {os.path.abspath(ref_path)}"
                logger.error(error_msg)
                self._is_initialized = False
                return False

            load_success = self._load_reference_image(ref_path)
            if not load_success:
                error_msg = f"❌ 灰度差分初始化失败：无法读取参考图像\n   路径: {ref_path}\n   请确认图像格式是否正确（支持 jpg/png/bmp）"
                logger.error(error_msg)
                self._is_initialized = False
                return False

            self._is_initialized = self._reference_image is not None

            if self._is_initialized:
                logger.info(f"✅ 灰度差分初始化成功")
                logger.info(f"   参考图像: {ref_path}")
                logger.info(f"   图像尺寸: {self._reference_image.shape[1]}x{self._reference_image.shape[0]}")
                logger.info(f"   差分阈值: {self._params['diff_threshold']}")
            else:
                logger.error("❌ 灰度差分初始化失败：参考图像加载后为空")
                return False

            return True
        except Exception as e:
            error_msg = f"❌ 灰度差分初始化异常：{str(e)}"
            logger.error(error_msg, exc_info=True)
            self._is_initialized = False
            return False

    def _load_reference_image(self, path: str) -> bool:
        try:
            ref_image = cv2.imread(path)
            if ref_image is None:
                logger.error(f"Failed to read reference image: {path}")
                return False

            if len(ref_image.shape) == 3:
                ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

            blur_kernel = self._params["blur_kernel"]
            if blur_kernel > 0:
                ref_image = cv2.GaussianBlur(ref_image, (blur_kernel, blur_kernel), 0)

            if self._params["normalize"]:
                ref_image = cv2.normalize(ref_image, None, 0, 255, cv2.NORM_MINMAX)

            self._reference_image = ref_image
            self._reference_image_path = path
            logger.info(f"Reference image loaded: {path}, shape: {ref_image.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            return False

    def set_reference_image(self, image: np.ndarray) -> bool:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            blur_kernel = self._params["blur_kernel"]
            if blur_kernel > 0:
                gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

            if self._params["normalize"]:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            self._reference_image = gray
            self._is_initialized = True
            logger.info(f"Reference image set manually, shape: {gray.shape}")
            return True
        except Exception as e:
            logger.error(f"Error setting reference image: {e}")
            return False

    def detect(self, image: np.ndarray, product_config: ProductConfig,
               roi: Optional[ROI] = None) -> Tuple[List[Defect], InferenceResult]:
        start_time = time.time()
        defects: List[Defect] = []

        try:
            if self._reference_image is None:
                error_msg = "Reference image not set for gray diff algorithm"
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
                gray = roi_image.copy()

            blur_kernel = self._params["blur_kernel"]
            if blur_kernel > 0:
                gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

            if self._params["normalize"]:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            ref_image = self._reference_image
            if ref_image.shape != gray.shape:
                if self._params["register_images"]:
                    ref_image = self._register_images(ref_image, gray)
                else:
                    ref_image = cv2.resize(ref_image, (gray.shape[1], gray.shape[0]))

            if self._params["use_abs_diff"]:
                diff = cv2.absdiff(gray, ref_image)
            else:
                diff = cv2.subtract(gray, ref_image)
                diff = np.maximum(diff, 0)

            if self._params["adaptive_threshold"]:
                diff_mask = cv2.adaptiveThreshold(
                    diff,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    self._params["adaptive_block_size"],
                    self._params["adaptive_C"]
                )
            else:
                _, diff_mask = cv2.threshold(
                    diff,
                    self._params["diff_threshold"],
                    255,
                    cv2.THRESH_BINARY
                )

            morph_kernel = self._params["morph_kernel"]
            if morph_kernel > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
                diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
                diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(
                diff_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            min_area = self._params["min_area_pixels"]
            max_area = self._params["max_area_pixels"]
            defect_type = self._params["defect_type"]
            diff_threshold = self._params["diff_threshold"]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                mask = np.zeros_like(diff)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_diff = cv2.mean(diff, mask=mask)[0]

                confidence = min(1.0, max(0.3, (mean_diff / 255.0) * 1.5))

                defect = self._create_defect(
                    contour, offset, defect_type, confidence, product_config
                )
                if defect:
                    if mean_diff > 0:
                        defect.description = (
                            f"{defect_type.value} detected, area: {defect.area_mm2:.4f} mm², "
                            f"mean diff: {mean_diff:.1f}"
                        )
                    defects.append(defect)

            inference_time = (time.time() - start_time) * 1000

            return defects, InferenceResult(
                success=True,
                inference_time_ms=inference_time,
                algorithm_type=self.algorithm_type,
                raw_output={
                    "contour_count": len(contours),
                    "defect_count": len(defects),
                    "max_diff": float(np.max(diff)),
                    "mean_diff": float(np.mean(diff))
                }
            )

        except Exception as e:
            logger.error(f"Gray diff detection failed: {e}", exc_info=True)
            inference_time = (time.time() - start_time) * 1000
            return defects, InferenceResult(
                success=False,
                inference_time_ms=inference_time,
                algorithm_type=self.algorithm_type,
                error_message=str(e)
            )

    def _register_images(self, ref_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        try:
            if ref_image.shape != target_image.shape:
                ref_image = cv2.resize(ref_image, (target_image.shape[1], target_image.shape[0]))

            warp_mode = cv2.MOTION_EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)

            (cc, warp_matrix) = cv2.findTransformECC(
                ref_image,
                target_image,
                warp_matrix,
                warp_mode,
                criteria
            )

            height, width = target_image.shape
            aligned_ref = cv2.warpAffine(
                ref_image,
                warp_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE
            )

            return aligned_ref
        except Exception as e:
            logger.warning(f"Image registration failed, using resize: {e}")
            return cv2.resize(ref_image, (target_image.shape[1], target_image.shape[0]))
