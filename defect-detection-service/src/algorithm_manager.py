from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import threading
import os
import yaml
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.algorithms.base_algorithm import BaseDetectionAlgorithm
from src.algorithms.edge_detection import EdgeDetectionAlgorithm
from src.algorithms.template_matching import TemplateMatchingAlgorithm
from src.algorithms.gray_diff import GrayDiffAlgorithm
from src.deep_learning.base_dl import BaseDeepLearningAlgorithm
from src.deep_learning.classification import ClassificationAlgorithm
from src.deep_learning.object_detection import ObjectDetectionAlgorithm
from src.deep_learning.segmentation import SegmentationAlgorithm
from src.utils.schemas import (
    ProductConfig, AlgorithmType, ImageData, DetectionOutput,
    DetectionResult, Defect, InferenceResult, AlertAction
)
from src.utils.logger import Logger

logger = Logger().logger


class AlgorithmManager:
    def __init__(self, enable_parallel: bool = True, max_workers: int = 4):
        self._traditional_algorithms: Dict[AlgorithmType, BaseDetectionAlgorithm] = {}
        self._dl_algorithms: Dict[AlgorithmType, BaseDeepLearningAlgorithm] = {}
        self._products: Dict[str, ProductConfig] = {}
        self._current_product_id: Optional[str] = None
        self._initialized_algorithms: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._is_initialized = False

        self._enable_parallel = enable_parallel
        self._max_workers = max_workers
        self._thread_pool: Optional[ThreadPoolExecutor] = None

        if self._enable_parallel:
            self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)
            logger.info(f"✅ 并行处理已启用，最大工作线程数: {max_workers}")

        self._register_algorithms()

    def _register_algorithms(self):
        self._traditional_algorithms[AlgorithmType.EDGE_DETECTION] = EdgeDetectionAlgorithm()
        self._traditional_algorithms[AlgorithmType.TEMPLATE_MATCHING] = TemplateMatchingAlgorithm()
        self._traditional_algorithms[AlgorithmType.GRAY_DIFF] = GrayDiffAlgorithm()

        self._dl_algorithms[AlgorithmType.CLASSIFICATION] = ClassificationAlgorithm()
        self._dl_algorithms[AlgorithmType.OBJECT_DETECTION] = ObjectDetectionAlgorithm()
        self._dl_algorithms[AlgorithmType.SEGMENTATION] = SegmentationAlgorithm()

        logger.info(f"Registered {len(self._traditional_algorithms)} traditional algorithms")
        logger.info(f"Registered {len(self._dl_algorithms)} deep learning algorithms")

    def load_products_config(self, config_path: str) -> bool:
        if not os.path.exists(config_path):
            logger.error(f"Products config file not found: {config_path}")
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            products_data = config_data.get("products", [])
            for product_data in products_data:
                product_config = self._parse_product_config(product_data)
                if product_config:
                    self._products[product_config.product_id] = product_config
                    logger.info(f"Loaded product config: {product_config.product_id} - {product_config.product_name}")

            self._is_initialized = len(self._products) > 0
            logger.info(f"Loaded {len(self._products)} product configurations")
            return self._is_initialized

        except Exception as e:
            logger.error(f"Failed to load products config: {e}", exc_info=True)
            return False

    def _parse_product_config(self, data: Dict[str, Any]) -> Optional[ProductConfig]:
        try:
            from src.utils.schemas import (
                ROI, AlgorithmConfig, DefectTypeConfig, DefectType,
                DefectSeverity, AlertAction, InferenceBackend, AlgorithmType
            )

            rois = []
            for roi_data in data.get("rois", []):
                rois.append(ROI(
                    x=roi_data.get("x", 0),
                    y=roi_data.get("y", 0),
                    width=roi_data.get("width", 0),
                    height=roi_data.get("height", 0),
                    enabled=roi_data.get("enabled", True),
                    name=roi_data.get("name", "")
                ))

            algorithms = []
            for algo_data in data.get("algorithms", []):
                algorithms.append(AlgorithmConfig(
                    type=AlgorithmType(algo_data.get("type", "edge_detection")),
                    enabled=algo_data.get("enabled", True),
                    params=algo_data.get("params", {}),
                    model_path=algo_data.get("model_path", "")
                ))

            defect_types = []
            for dt_data in data.get("defect_types", []):
                defect_types.append(DefectTypeConfig(
                    type=DefectType(dt_data.get("type", "unknown")),
                    enabled=dt_data.get("enabled", True),
                    min_area_mm2=dt_data.get("min_area_mm2", 0.0),
                    max_area_mm2=dt_data.get("max_area_mm2", float('inf')),
                    min_confidence=dt_data.get("min_confidence", 0.5),
                    severity=DefectSeverity(dt_data.get("severity", "major")),
                    alert_action=AlertAction(dt_data.get("alert_action", "reject"))
                ))

            return ProductConfig(
                product_id=data.get("product_id", ""),
                product_name=data.get("product_name", ""),
                pixel_to_mm_ratio=data.get("pixel_to_mm_ratio", 0.01),
                rois=rois,
                algorithms=algorithms,
                defect_types=defect_types,
                sensitivity=data.get("sensitivity", 0.8),
                allowed_error_mm=data.get("allowed_error_mm", 0.1),
                allow_multiple_defects=data.get("allow_multiple_defects", False),
                max_defects_allowed=data.get("max_defects_allowed", 0),
                inference_backend=InferenceBackend(data.get("inference_backend", "onnx_cpu")),
                gpu_device_id=data.get("gpu_device_id", 0),
                enable_tensorrt=data.get("enable_tensorrt", False),
                inference_timeout_ms=data.get("inference_timeout_ms", 100)
            )

        except Exception as e:
            logger.error(f"Failed to parse product config: {e}", exc_info=True)
            return None

    def set_current_product(self, product_id: str) -> bool:
        with self._lock:
            if product_id not in self._products:
                logger.error(f"Product not found: {product_id}")
                return False

            self._current_product_id = product_id
            product_config = self._products[product_id]

            logger.info(f"Switching to product: {product_config.product_id} - {product_config.product_name}")

            self._cleanup_unused_algorithms(product_config)

            for algo_config in product_config.algorithms:
                if not algo_config.enabled:
                    continue

                algo_key = f"{product_id}_{algo_config.type.value}"
                if algo_key in self._initialized_algorithms:
                    continue

                self._initialize_algorithm(algo_config, product_config, algo_key)

            return True

    def _initialize_algorithm(self, algo_config: Any, product_config: ProductConfig, algo_key: str):
        try:
            algo_type = algo_config.type

            if algo_type in self._traditional_algorithms:
                algo = self._traditional_algorithms[algo_type]
                if not algo.is_initialized:
                    if algo.initialize(algo_config.params):
                        self._initialized_algorithms[algo_key] = algo
                        logger.info(f"Initialized traditional algorithm: {algo_type.value}")
                    else:
                        logger.error(f"Failed to initialize traditional algorithm: {algo_type.value}")
                else:
                    self._initialized_algorithms[algo_key] = algo

            elif algo_type in self._dl_algorithms:
                algo = self._dl_algorithms[algo_type]
                if not algo.is_initialized:
                    model_path = algo_config.model_path
                    if not model_path or not os.path.exists(model_path):
                        logger.warning(f"Model path not found for {algo_type.value}, skipping")
                        return

                    if algo.initialize(
                        params=algo_config.params,
                        model_path=model_path,
                        backend=product_config.inference_backend.value,
                        gpu_device_id=product_config.gpu_device_id,
                        enable_tensorrt=product_config.enable_tensorrt
                    ):
                        self._initialized_algorithms[algo_key] = algo
                        logger.info(f"Initialized DL algorithm: {algo_type.value} with model: {model_path}")
                    else:
                        logger.error(f"Failed to initialize DL algorithm: {algo_type.value}")
                else:
                    self._initialized_algorithms[algo_key] = algo

        except Exception as e:
            logger.error(f"Error initializing algorithm {algo_config.type.value}: {e}", exc_info=True)

    def _cleanup_unused_algorithms(self, current_product: ProductConfig):
        active_algos = {
            f"{current_product.product_id}_{algo.type.value}"
            for algo in current_product.algorithms if algo.enabled
        }

        for key in list(self._initialized_algorithms.keys()):
            if key not in active_algos:
                algo = self._initialized_algorithms.pop(key)
                if hasattr(algo, 'destroy'):
                    try:
                        algo.destroy()
                    except Exception as e:
                        logger.warning(f"Error destroying algorithm: {e}")
                logger.info(f"Cleaned up algorithm: {key}")

    def detect(self, image_data: ImageData) -> DetectionOutput:
        with self._lock:
            if self._current_product_id is None:
                logger.error("No product selected, call set_current_product first")
                return DetectionOutput.create(
                    sequence_id=image_data.sequence_id,
                    product_id="unknown",
                    result=DetectionResult.NG
                )

            product_config = self._products[self._current_product_id]

            output = DetectionOutput.create(
                sequence_id=image_data.sequence_id,
                product_id=product_config.product_id,
                image_data=image_data,
                line_id=image_data.metadata.get("line_id", ""),
                station_id=image_data.metadata.get("station_id", "")
            )

            all_defects: List[Defect] = []
            algorithm_times: Dict[str, float] = {}
            total_start_time = time.time()

            rois = product_config.rois
            if not rois:
                rois = [None]

            enabled_algos = []
            for algo_config in product_config.algorithms:
                if not algo_config.enabled:
                    continue
                algo_key = f"{product_config.product_id}_{algo_config.type.value}"
                if algo_key not in self._initialized_algorithms:
                    logger.warning(f"Algorithm {algo_config.type.value} not initialized, skipping")
                    continue
                enabled_algos.append((algo_config, algo_key))

            if self._enable_parallel and self._thread_pool and len(enabled_algos) > 0:
                all_defects, algorithm_times = self._detect_parallel(
                    image_data.image, product_config, enabled_algos, rois
                )
            else:
                all_defects, algorithm_times = self._detect_sequential(
                    image_data.image, product_config, enabled_algos, rois
                )

            all_defects = self._deduplicate_defects(all_defects, product_config)

            output.defects = all_defects
            output.total_inference_time_ms = (time.time() - total_start_time) * 1000
            output.algorithm_times = algorithm_times

            output.result = self._determine_result(all_defects, product_config)
            output.alert_action = self._determine_alert_action(all_defects, product_config)

            return output

    def _detect_sequential(self, image: np.ndarray, product_config: ProductConfig,
                           enabled_algos: List[Tuple[Any, str]], rois: List[Optional[ROI]]
                           ) -> Tuple[List[Defect], Dict[str, float]]:
        all_defects: List[Defect] = []
        algorithm_times: Dict[str, float] = {}

        for algo_config, algo_key in enabled_algos:
            algo = self._initialized_algorithms[algo_key]

            for roi in rois:
                try:
                    defects, algo_time = self._run_single_detection(
                        algo, algo_config, image, product_config, roi
                    )
                    if defects is not None:
                        all_defects.extend(defects)
                        algo_type_str = algo_config.type.value
                        if roi and roi.name:
                            algo_type_str += f"_{roi.name}"
                        algorithm_times[algo_type_str] = algo_time
                except Exception as e:
                    logger.error(f"Error running algorithm {algo_config.type.value}: {e}", exc_info=True)

        return all_defects, algorithm_times

    def _detect_parallel(self, image: np.ndarray, product_config: ProductConfig,
                         enabled_algos: List[Tuple[Any, str]], rois: List[Optional[ROI]]
                         ) -> Tuple[List[Defect], Dict[str, float]]:
        all_defects: List[Defect] = []
        algorithm_times: Dict[str, float] = {}

        tasks = []
        for algo_config, algo_key in enabled_algos:
            algo = self._initialized_algorithms[algo_key]
            for roi in rois:
                tasks.append((algo, algo_config, roi))

        logger.debug(f"🚀 并行执行 {len(tasks)} 个检测任务 (算法×ROI)")

        futures = []
        for algo, algo_config, roi in tasks:
            future = self._thread_pool.submit(
                self._run_single_detection,
                algo, algo_config, image, product_config, roi
            )
            futures.append((future, algo_config, roi))

        timeout_sec = product_config.inference_timeout_ms / 1000.0

        for future, algo_config, roi in futures:
            try:
                defects, algo_time = future.result(timeout=timeout_sec)
                if defects is not None:
                    all_defects.extend(defects)
                    algo_type_str = algo_config.type.value
                    if roi and roi.name:
                        algo_type_str += f"_{roi.name}"
                    algorithm_times[algo_type_str] = algo_time
            except FuturesTimeoutError:
                algo_type_str = algo_config.type.value
                if roi and roi.name:
                    algo_type_str += f"_{roi.name}"
                logger.error(f"⏰ 检测超时 [{algo_type_str}]: 超过 {product_config.inference_timeout_ms}ms")
                future.cancel()
            except Exception as e:
                algo_type_str = algo_config.type.value
                if roi and roi.name:
                    algo_type_str += f"_{roi.name}"
                logger.error(f"❌ 并行检测错误 [{algo_type_str}]: {e}", exc_info=True)

        return all_defects, algorithm_times

    def _run_single_detection(self, algo, algo_config, image: np.ndarray,
                              product_config: ProductConfig, roi: Optional[ROI]
                              ) -> Tuple[Optional[List[Defect]], float]:
        algo_start_time = time.time()

        try:
            defects, infer_result = algo.detect(
                image=image,
                product_config=product_config,
                roi=roi
            )

            algo_time = (time.time() - algo_start_time) * 1000

            if infer_result.success:
                return defects, algo_time
            else:
                logger.warning(f"算法 {algo_config.type.value} 检测失败: {infer_result.error_message}")
                return None, algo_time

        except Exception as e:
            algo_time = (time.time() - algo_start_time) * 1000
            logger.error(f"算法 {algo_config.type.value} 检测异常: {e}", exc_info=True)
            return None, algo_time

    def _deduplicate_defects(self, defects: List[Defect], product_config: ProductConfig) -> List[Defect]:
        if len(defects) <= 1:
            return defects

        iou_threshold = product_config.allowed_error_mm
        pixel_ratio = product_config.pixel_to_mm_ratio
        iou_threshold_pixels = iou_threshold / pixel_ratio if pixel_ratio > 0 else 10

        keep = [True] * len(defects)

        for i in range(len(defects)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(defects)):
                if not keep[j]:
                    continue

                iou = self._calculate_iou(defects[i].bbox, defects[j].bbox)
                if iou > 0.3:
                    if defects[i].confidence >= defects[j].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [defects[i] for i in range(len(defects)) if keep[i]]

    def _calculate_iou(self, bbox1: Any, bbox2: Any) -> float:
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection

        return intersection / union if union > 0 else 0.0

    def _determine_result(self, defects: List[Defect], product_config: ProductConfig) -> DetectionResult:
        if not defects:
            return DetectionResult.OK

        critical_defects = [d for d in defects if d.severity.value == "critical"]
        if critical_defects:
            return DetectionResult.NG

        if not product_config.allow_multiple_defects:
            return DetectionResult.NG

        if product_config.max_defects_allowed > 0 and len(defects) > product_config.max_defects_allowed:
            return DetectionResult.NG

        return DetectionResult.OK

    def _determine_alert_action(self, defects: List[Defect], product_config: ProductConfig) -> AlertAction:
        if not defects:
            return AlertAction.NONE

        max_action = AlertAction.NONE

        for defect in defects:
            defect_config = product_config.get_defect_config(defect.type)
            if defect_config:
                action = defect_config.alert_action
            else:
                if defect.severity.value == "critical":
                    action = AlertAction.STOP_LINE
                elif defect.severity.value == "major":
                    action = AlertAction.REJECT
                elif defect.severity.value == "minor":
                    action = AlertAction.LOG
                else:
                    action = AlertAction.WARN

            action_priority = {
                AlertAction.NONE: 0,
                AlertAction.LOG: 1,
                AlertAction.WARN: 2,
                AlertAction.REJECT: 3,
                AlertAction.STOP_LINE: 4
            }

            if action_priority.get(action, 0) > action_priority.get(max_action, 0):
                max_action = action

        return max_action

    def get_product_config(self, product_id: str = None) -> Optional[ProductConfig]:
        if product_id is None:
            product_id = self._current_product_id

        return self._products.get(product_id)

    def get_all_products(self) -> Dict[str, ProductConfig]:
        return self._products.copy()

    def get_available_algorithms(self) -> Dict[str, List[str]]:
        return {
            "traditional": [algo.value for algo in self._traditional_algorithms.keys()],
            "deep_learning": [algo.value for algo in self._dl_algorithms.keys()]
        }

    def reload_algorithm_params(self, algo_type: AlgorithmType, params: Dict[str, Any]) -> bool:
        with self._lock:
            if self._current_product_id is None:
                return False

            product_config = self._products[self._current_product_id]
            algo_config = product_config.get_algorithm_config(algo_type)

            if algo_config is None:
                return False

            algo_config.params.update(params)

            algo_key = f"{product_config.product_id}_{algo_type.value}"
            if algo_key in self._initialized_algorithms:
                algo = self._initialized_algorithms[algo_key]
                for key, value in params.items():
                    algo.set_param(key, value)

            logger.info(f"Updated params for {algo_type.value}: {params}")
            return True

    def cleanup(self):
        with self._lock:
            for algo in self._initialized_algorithms.values():
                if hasattr(algo, 'destroy'):
                    try:
                        algo.destroy()
                    except Exception as e:
                        logger.warning(f"Error during cleanup: {e}")

            self._initialized_algorithms.clear()

            if self._thread_pool:
                try:
                    self._thread_pool.shutdown(wait=True, timeout=5)
                    logger.info("线程池已关闭")
                except Exception as e:
                    logger.warning(f"关闭线程池异常: {e}")
                self._thread_pool = None

            logger.info("All algorithms cleaned up")

    @property
    def current_product_id(self) -> Optional[str]:
        return self._current_product_id

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
