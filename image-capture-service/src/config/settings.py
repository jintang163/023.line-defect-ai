import yaml
import os
from typing import Dict, List, Any
from src.utils.logger import Logger
from src.utils.schemas import (
    CameraConfig, LightChannelConfig, ROI, DistortionParams,
    CameraType, LightMode, TriggerMode
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

    def get_camera_configs(self) -> List[CameraConfig]:
        camera_configs = []
        for cam in self._config.get("cameras", []):
            if not cam.get("enabled", True):
                continue
            camera_configs.append(CameraConfig(
                id=cam["id"],
                name=cam["name"],
                type=CameraType(cam["type"]),
                position=cam["position"],
                ip=cam["ip"],
                serial_number=cam["serial_number"],
                exposure_time=cam["exposure_time"],
                gain=cam["gain"],
                width=cam["width"],
                height=cam["height"],
                pixel_format=cam["pixel_format"],
                trigger_mode=TriggerMode(cam["trigger_mode"]),
                enabled=cam.get("enabled", True)
            ))
        return camera_configs

    def get_light_channels(self) -> List[LightChannelConfig]:
        lighting = self._config.get("lighting", {})
        channels = []
        for ch in lighting.get("channels", []):
            channels.append(LightChannelConfig(
                id=ch["id"],
                name=ch["name"],
                mode=LightMode(ch["mode"]),
                brightness=ch["brightness"],
                color_temp=ch["color_temp"],
                strobe_delay=ch.get("strobe_delay", 0),
                strobe_width=ch.get("strobe_width", 1000)
            ))
        return channels

    def get_lighting_config(self) -> Dict[str, Any]:
        return self._config.get("lighting", {})

    def get_preprocessing_config(self) -> Dict[str, Any]:
        return self._config.get("preprocessing", {})

    def get_roi(self) -> ROI:
        roi_cfg = self.get_preprocessing_config().get("roi", {})
        return ROI(
            x=roi_cfg.get("x", 0),
            y=roi_cfg.get("y", 0),
            width=roi_cfg.get("width", 0),
            height=roi_cfg.get("height", 0)
        )

    def get_distortion_params(self) -> DistortionParams:
        params = self.get_preprocessing_config().get("distortion_params", {})
        return DistortionParams.from_config(params)

    def get_buffer_config(self) -> Dict[str, Any]:
        return self._config.get("buffer", {})

    def get_messaging_config(self) -> Dict[str, Any]:
        return self._config.get("messaging", {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        return self._config.get("monitoring", {})

    def get_trigger_config(self) -> Dict[str, Any]:
        return self._config.get("trigger", {})
