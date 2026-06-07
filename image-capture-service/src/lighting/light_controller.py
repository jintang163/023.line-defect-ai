import time
import threading
from typing import Dict, Optional
from src.lighting.base_controller import BaseLightController
from src.utils.schemas import LightChannelConfig, LightMode
from src.utils.logger import Logger

logger = Logger().logger


class ModbusLightController(BaseLightController):
    def __init__(self, channels, host: str = "192.168.1.200", port: int = 502):
        super().__init__(channels)
        self.host = host
        self.port = port
        self._client = None
        self._strobe_timers: Dict[str, threading.Timer] = {}
        self._channel_states: Dict[str, dict] = {}

        for ch in channels:
            self._channel_states[ch.id] = {
                "brightness": ch.brightness,
                "mode": ch.mode,
                "color_temp": ch.color_temp,
                "is_on": ch.mode != LightMode.OFF
            }

    def _try_import_modbus(self) -> bool:
        try:
            from pymodbus.client import ModbusTcpClient
            self._client_class = ModbusTcpClient
            return True
        except ImportError:
            logger.warning("pymodbus not installed, using mock mode")
            return False

    def connect(self) -> bool:
        try:
            if self._try_import_modbus():
                self._client = self._client_class(host=self.host, port=self.port)
                if not self._client.connect():
                    raise ConnectionError(f"Failed to connect to Modbus server at {self.host}:{self.port}")

            self._is_connected = True
            logger.info(f"Light controller connected to {self.host}:{self.port}")

            for ch_id, config in self.channels.items():
                self.set_brightness(ch_id, config.brightness)
                self.set_mode(ch_id, config.mode)
                self.set_color_temp(ch_id, config.color_temp)

            return True
        except Exception as e:
            logger.error(f"Failed to connect light controller: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            for timer in self._strobe_timers.values():
                timer.cancel()
            self._strobe_timers.clear()

            if self._client:
                self._client.close()
                self._client = None

            self._is_connected = False
            logger.info("Light controller disconnected")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect light controller: {e}")
            return False

    def set_brightness(self, channel_id: str, brightness: int) -> bool:
        if not self._is_connected:
            return False
        if channel_id not in self.channels:
            return False

        brightness = max(0, min(100, brightness))

        try:
            if self._client:
                register_addr = self._get_brightness_register(channel_id)
                self._client.write_register(register_addr, brightness)

            self._channel_states[channel_id]["brightness"] = brightness
            self.channels[channel_id].brightness = brightness
            logger.debug(f"Channel {channel_id} brightness set to {brightness}%")
            return True
        except Exception as e:
            logger.error(f"Failed to set brightness for {channel_id}: {e}")
            return False

    def set_mode(self, channel_id: str, mode: LightMode) -> bool:
        if not self._is_connected:
            return False
        if channel_id not in self.channels:
            return False

        try:
            if self._client:
                register_addr = self._get_mode_register(channel_id)
                mode_value = {"off": 0, "continuous": 1, "strobe": 2}[mode.value]
                self._client.write_register(register_addr, mode_value)

            self._channel_states[channel_id]["mode"] = mode
            self._channel_states[channel_id]["is_on"] = mode != LightMode.OFF
            self.channels[channel_id].mode = mode
            logger.debug(f"Channel {channel_id} mode set to {mode.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set mode for {channel_id}: {e}")
            return False

    def set_color_temp(self, channel_id: str, color_temp: int) -> bool:
        if not self._is_connected:
            return False
        if channel_id not in self.channels:
            return False

        color_temp = max(2700, min(6500, color_temp))

        try:
            if self._client:
                register_addr = self._get_color_temp_register(channel_id)
                self._client.write_register(register_addr, color_temp)

            self._channel_states[channel_id]["color_temp"] = color_temp
            self.channels[channel_id].color_temp = color_temp
            logger.debug(f"Channel {channel_id} color temp set to {color_temp}K")
            return True
        except Exception as e:
            logger.error(f"Failed to set color temp for {channel_id}: {e}")
            return False

    def trigger_strobe(self, channel_id: str, delay_us: int = 0, width_us: int = 1000) -> bool:
        if not self._is_connected:
            return False
        if channel_id not in self.channels:
            return False

        try:
            state = self._channel_states[channel_id]
            if state["mode"] != LightMode.STROBE:
                return True

            if channel_id in self._strobe_timers:
                self._strobe_timers[channel_id].cancel()

            if self._client:
                trigger_addr = self._get_strobe_register(channel_id)
                self._client.write_register(trigger_addr, 1)

            state["is_on"] = True
            logger.debug(f"Strobe triggered on channel {channel_id}, width: {width_us}us")

            def _turn_off():
                state["is_on"] = False
                if self._client:
                    trigger_addr = self._get_strobe_register(channel_id)
                    self._client.write_register(trigger_addr, 0)

            timer = threading.Timer(width_us / 1_000_000.0, _turn_off)
            timer.start()
            self._strobe_timers[channel_id] = timer

            return True
        except Exception as e:
            logger.error(f"Failed to trigger strobe for {channel_id}: {e}")
            return False

    def _get_brightness_register(self, channel_id: str) -> int:
        mapping = {"ch-top": 100, "ch-bottom": 101, "ch-side": 102}
        return mapping.get(channel_id, 100)

    def _get_mode_register(self, channel_id: str) -> int:
        mapping = {"ch-top": 200, "ch-bottom": 201, "ch-side": 202}
        return mapping.get(channel_id, 200)

    def _get_color_temp_register(self, channel_id: str) -> int:
        mapping = {"ch-top": 300, "ch-bottom": 301, "ch-side": 302}
        return mapping.get(channel_id, 300)

    def _get_strobe_register(self, channel_id: str) -> int:
        mapping = {"ch-top": 400, "ch-bottom": 401, "ch-side": 402}
        return mapping.get(channel_id, 400)

    def get_channel_state(self, channel_id: str) -> Optional[dict]:
        return self._channel_states.get(channel_id)
