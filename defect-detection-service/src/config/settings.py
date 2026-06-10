import yaml
import os
from typing import Dict, List, Any, Optional
from src.utils.logger import Logger
from src.utils.schemas import (
    InferenceBackend, AlertAction
)

logger = Logger().logger


class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path or "./config/config.yaml")
        return cls._instance

    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        logger.info(f"Config loaded from {config_path}")

    def reload(self):
        self._load_config("./config/config.yaml")
        logger.info("Config reloaded")

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get_service_config(self) -> Dict[str, Any]:
        return self._config.get("service", {})

    def get_messaging_config(self) -> Dict[str, Any]:
        return self._config.get("messaging", {})

    def get_products_config_path(self) -> str:
        return self._config.get("detection", {}).get("products_config", "./config/products.yaml")

    def get_models_dir(self) -> str:
        return self._config.get("detection", {}).get("models_dir", "./models")

    def get_default_inference_backend(self) -> InferenceBackend:
        backend_str = self._config.get("detection", {}).get("default_backend", "onnx_cpu")
        return InferenceBackend(backend_str)

    def get_gpu_device_id(self) -> int:
        return self._config.get("detection", {}).get("gpu_device_id", 0)

    def get_enable_tensorrt(self) -> bool:
        return self._config.get("detection", {}).get("enable_tensorrt", False)

    def get_inference_timeout_ms(self) -> int:
        return self._config.get("detection", {}).get("inference_timeout_ms", 100)

    def get_consecutive_ng_threshold(self) -> int:
        return self._config.get("alert", {}).get("consecutive_ng_threshold", 5)

    def get_alert_webhook_url(self) -> str:
        return self._config.get("alert", {}).get("webhook_url", "")

    def get_plc_config(self) -> Dict[str, Any]:
        return self._config.get("plc", {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        return self._config.get("monitoring", {})

    def get_api_config(self) -> Dict[str, Any]:
        return self._config.get("api", {})

    def get_auto_stop_line(self) -> bool:
        return self._config.get("alert", {}).get("auto_stop_line", True)

    def get_action_log_config(self) -> Dict[str, Any]:
        return self._config.get("action_log", {})

    def get_production_config(self) -> Dict[str, Any]:
        return self._config.get("production", {})

    def get_manual_override_config(self) -> Dict[str, Any]:
        return self._config.get("manual_override", {})

    def get_data_management_config(self) -> Dict[str, Any]:
        return self._config.get("data_management", {})

    def get_system_monitor_config(self) -> Dict[str, Any]:
        return self._config.get("system_monitor", {})

    def get_notification_config(self) -> Dict[str, Any]:
        return self._config.get("notification", {})

    def get_alert_config(self) -> Dict[str, Any]:
        return self._config.get("alert", {})

    def get_model_management_config(self) -> Dict[str, Any]:
        return self._config.get("model_management", {})
