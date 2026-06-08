from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import time
import os
import threading

from src.utils.schemas import InferenceBackend
from src.utils.logger import Logger

logger = Logger().logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available, deep learning inference will be disabled")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available, GPU acceleration limited")


class ONNXInferenceEngine:
    def __init__(self, model_path: str, backend: InferenceBackend = InferenceBackend.ONNX_CPU,
                 gpu_device_id: int = 0, enable_tensorrt: bool = False,
                 enable_dynamic_batch: bool = False, max_batch_size: int = 16):
        self.model_path = model_path
        self.backend = backend
        self.gpu_device_id = gpu_device_id
        self.enable_tensorrt = enable_tensorrt
        self.enable_dynamic_batch = enable_dynamic_batch
        self.max_batch_size = max_batch_size

        self._session: Optional["ort.InferenceSession"] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, List[int]] = {}
        self._is_initialized = False
        self._lock = threading.Lock()

        self._tensorrt_engine = None
        self._tensorrt_context = None
        self._cuda_inputs = []
        self._cuda_outputs = []
        self._cuda_bindings = []
        self._cuda_stream = None

    def initialize(self) -> bool:
        with self._lock:
            if self._is_initialized:
                return True

            if not ONNX_AVAILABLE:
                logger.error("onnxruntime not installed, cannot initialize inference engine")
                return False

            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            try:
                if self.enable_tensorrt and TENSORRT_AVAILABLE:
                    return self._initialize_tensorrt()
                else:
                    return self._initialize_onnx()
            except Exception as e:
                logger.error(f"Failed to initialize inference engine: {e}", exc_info=True)
                self._is_initialized = False
                return False

    def _initialize_onnx(self) -> bool:
        logger.info(f"Initializing ONNX Runtime engine with backend: {self.backend.value}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if self.backend == InferenceBackend.ONNX_GPU:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                logger.warning("CUDA not available, falling back to CPU")
                providers = ['CPUExecutionProvider']
            else:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': self.gpu_device_id,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
        else:
            providers = ['CPUExecutionProvider']

        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        self._input_names = [input.name for input in self._session.get_inputs()]
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._input_shapes = {
            input.name: list(input.shape) for input in self._session.get_inputs()
        }

        logger.info(f"ONNX Runtime engine initialized successfully")
        logger.info(f"Inputs: {self._input_names}, Outputs: {self._output_names}")

        self._is_initialized = True
        return True

    def _initialize_tensorrt(self) -> bool:
        logger.info(f"Initializing TensorRT engine for model: {self.model_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)

        with open(self.model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                return False

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30

        if self.enable_dynamic_batch:
            profile = builder.create_optimization_profile()
            for input in network:
                input_shape = input.shape
                min_shape = (1,) + tuple(input_shape[1:])
                opt_shape = (self.max_batch_size // 2,) + tuple(input_shape[1:])
                max_shape = (self.max_batch_size,) + tuple(input_shape[1:])
                profile.set_shape(input.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        runtime = trt.Runtime(trt_logger)
        self._tensorrt_engine = runtime.deserialize_cuda_engine(serialized_engine)
        self._tensorrt_context = self._tensorrt_engine.create_execution_context()

        self._cuda_inputs = []
        self._cuda_outputs = []
        self._cuda_bindings = []

        for binding in self._tensorrt_engine:
            size = trt.volume(self._tensorrt_engine.get_binding_shape(binding))
            dtype = trt.nptype(self._tensorrt_engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._cuda_bindings.append(int(device_mem))

            if self._tensorrt_engine.binding_is_input(binding):
                self._cuda_inputs.append((host_mem, device_mem))
                self._input_names.append(binding)
            else:
                self._cuda_outputs.append((host_mem, device_mem))
                self._output_names.append(binding)

        self._cuda_stream = cuda.Stream()

        logger.info("TensorRT engine initialized successfully")
        self._is_initialized = True
        return True

    def infer(self, inputs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        start_time = time.time()

        if not self._is_initialized:
            raise RuntimeError("Inference engine not initialized")

        with self._lock:
            try:
                if self._tensorrt_engine is not None:
                    outputs = self._infer_tensorrt(inputs)
                else:
                    outputs = self._infer_onnx(inputs)

                inference_time = (time.time() - start_time) * 1000
                return outputs, inference_time

            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                raise

    def _infer_onnx(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        onnx_inputs = {}
        for name, arr in inputs.items():
            if name in self._input_names:
                onnx_inputs[name] = arr.astype(np.float32)

        outputs = self._session.run(self._output_names, onnx_inputs)
        return {name: output for name, output in zip(self._output_names, outputs)}

    def _infer_tensorrt(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for i, (name, arr) in enumerate(inputs.items()):
            if i < len(self._cuda_inputs):
                host_mem, device_mem = self._cuda_inputs[i]
                np.copyto(host_mem, arr.ravel().astype(np.float32))
                cuda.memcpy_htod_async(device_mem, host_mem, self._cuda_stream)

        self._tensorrt_context.execute_async_v2(
            bindings=self._cuda_bindings,
            stream_handle=self._cuda_stream.handle
        )

        outputs = {}
        for i, (name, (host_mem, device_mem)) in enumerate(zip(self._output_names, self._cuda_outputs)):
            cuda.memcpy_dtoh_async(host_mem, device_mem, self._cuda_stream)
            self._cuda_stream.synchronize()

            binding_idx = len(self._cuda_inputs) + i
            shape = self._tensorrt_engine.get_binding_shape(binding_idx)
            outputs[name] = host_mem.reshape(tuple(shape))

        return outputs

    def run_batch(self, batch_inputs: List[Dict[str, np.ndarray]]) -> Tuple[List[Dict[str, np.ndarray]], float]:
        if not self.enable_dynamic_batch or len(batch_inputs) <= 1:
            results = []
            total_time = 0.0
            for inputs in batch_inputs:
                outputs, infer_time = self.infer(inputs)
                results.append(outputs)
                total_time += infer_time
            return results, total_time

        start_time = time.time()

        batched_inputs = {}
        for name in self._input_names:
            tensors = [inputs[name] for inputs in batch_inputs if name in inputs]
            if tensors:
                batched_inputs[name] = np.stack(tensors, axis=0)

        outputs, _ = self.infer(batched_inputs)

        batch_size = len(batch_inputs)
        results = []
        for i in range(batch_size):
            result = {}
            for name, output in outputs.items():
                result[name] = output[i:i+1]
            results.append(result)

        total_time = (time.time() - start_time) * 1000
        return results, total_time

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int],
                         mean: List[float] = None, std: List[float] = None,
                         normalize: bool = True, to_chw: bool = True) -> np.ndarray:
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        img_array = resized.astype(np.float32)

        if normalize:
            img_array /= 255.0
            img_array -= np.array(mean, dtype=np.float32)
            img_array /= np.array(std, dtype=np.float32)

        if to_chw:
            img_array = np.transpose(img_array, (2, 0, 1))

        return img_array

    def get_input_shape(self, input_name: str = None) -> List[int]:
        if input_name is None and self._input_names:
            input_name = self._input_names[0]
        return self._input_shapes.get(input_name, [])

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def input_names(self) -> List[str]:
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        return self._output_names

    def destroy(self):
        with self._lock:
            if self._cuda_stream is not None:
                self._cuda_stream = None
            if self._tensorrt_context is not None:
                self._tensorrt_context = None
            if self._tensorrt_engine is not None:
                self._tensorrt_engine = None

            for _, device_mem in self._cuda_inputs + self._cuda_outputs:
                if device_mem is not None:
                    device_mem.free()

            self._cuda_inputs = []
            self._cuda_outputs = []
            self._cuda_bindings = []
            self._session = None
            self._is_initialized = False


import cv2
