from abc import ABC, abstractmethod
import threading
import time
from typing import Dict, List, Optional
from src.utils.schemas import LightChannelConfig, LightMode
from src.utils.logger import Logger

logger = Logger().logger


class BaseLightController(ABC):
    def __init__(self, channels: List[LightChannelConfig]):
        self.channels: Dict[str, LightChannelConfig] = {ch.id: ch for ch in channels}
        self._lock = threading.Lock()
        self._is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def set_brightness(self, channel_id: str, brightness: int) -> bool:
        pass

    @abstractmethod
    def set_mode(self, channel_id: str, mode: LightMode) -> bool:
        pass

    @abstractmethod
    def set_color_temp(self, channel_id: str, color_temp: int) -> bool:
        pass

    @abstractmethod
    def trigger_strobe(self, channel_id: str, delay_us: int = 0, width_us: int = 1000) -> bool:
        pass

    def is_connected(self) -> bool:
        return self._is_connected

    def get_channel_config(self, channel_id: str) -> Optional[LightChannelConfig]:
        return self.channels.get(channel_id)

    def get_all_channels(self) -> List[LightChannelConfig]:
        return list(self.channels.values())

    def trigger_all_strobe(self, delay_us: int = 0, width_us: int = 1000) -> bool:
        success = True
        for channel_id in self.channels:
            config = self.channels[channel_id]
            if config.mode == LightMode.STROBE:
                if not self.trigger_strobe(channel_id, config.strobe_delay, config.strobe_width):
                    success = False
        return success

    def set_all_brightness(self, brightness: int) -> Dict[str, bool]:
        results = {}
        for channel_id in self.channels:
            results[channel_id] = self.set_brightness(channel_id, brightness)
        return results

    def apply_material_preset(self, material: str) -> bool:
        presets = {
            "metal": {
                "mode": LightMode.STROBE,
                "brightness": 85,
                "color_temp": 6500
            },
            "glass": {
                "mode": LightMode.CONTINUOUS,
                "brightness": 60,
                "color_temp": 5500
            },
            "plastic": {
                "mode": LightMode.STROBE,
                "brightness": 75,
                "color_temp": 5000
            }
        }

        preset = presets.get(material.lower())
        if not preset:
            logger.warning(f"Unknown material preset: {material}")
            return False

        logger.info(f"Applying lighting preset for material: {material}")
        for channel_id in self.channels:
            self.set_mode(channel_id, preset["mode"])
            self.set_brightness(channel_id, preset["brightness"])
            self.set_color_temp(channel_id, preset["color_temp"])

        return True
