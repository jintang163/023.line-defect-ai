import time
import threading
from typing import Dict, Optional
from src.lighting.base_controller import BaseLightController
from src.utils.schemas import LightChannelConfig, LightMode
from src.utils.logger import Logger

logger = Logger().logger


class ModbusLightController(BaseLightController):
    def __init__(self, channels, host: str = "192.168.1.200", port: int = 502,
                 allow_mock_fallback: bool = False):
        super().__init__(channels)
        self.host = host
        self.port = port
        self._client = None
        self._strobe_timers: Dict[str, threading.Timer] = {}
        self._channel_states: Dict[str, dict] = {}
        self._modbus_available = False
        self._hardware_ready = False
        self._allow_mock_fallback = allow_mock_fallback
        self._mock_mode_active = False

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
            self._modbus_available = True
            return True
        except ImportError:
            self._modbus_available = False
            logger.error(
                "pymodbus not installed! Light controller cannot communicate with hardware. "
                "Install with: pip install pymodbus"
            )
            return False

    def connect(self) -> bool:
        if not self._try_import_modbus():
            if self._allow_mock_fallback:
                logger.warning(
                    "pymodbus not available, falling back to MOCK mode. "
                    "THIS IS FOR DEVELOPMENT ONLY - NO HARDWARE CONTROL!"
                )
                self._mock_mode_active = True
                self._is_connected = False
                return False
            logger.error("Modbus light controller connection failed: pymodbus not installed")
            return False

        try:
            self._client = self._client_class(host=self.host, port=self.port, timeout=5)

            if not self._client.connect():
                raise ConnectionError(
                    f"Failed to connect to Modbus light controller at {self.host}:{self.port}. "
                    f"Check: 1) Controller power, 2) Network connection, "
                    f"3) IP address and port configuration."
                )

            if not self._verify_hardware():
                self._client.close()
                raise ConnectionError(
                    f"Light controller at {self.host}:{self.port} did not respond correctly. "
                    f"Check Modbus register map configuration."
                )

            self._is_connected = True
            self._hardware_ready = True
            self._mock_mode_active = False
            logger.info(f"Light controller successfully connected to {self.host}:{self.port}")

            for ch_id, config in self.channels.items():
                if not self.set_brightness(ch_id, config.brightness):
                    logger.warning(f"Failed to set initial brightness for channel {ch_id}")
                if not self.set_mode(ch_id, config.mode):
                    logger.warning(f"Failed to set initial mode for channel {ch_id}")
                if not self.set_color_temp(ch_id, config.color_temp):
                    logger.warning(f"Failed to set initial color temp for channel {ch_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect light controller: {e}")
            self._is_connected = False
            self._hardware_ready = False
            self._client = None

            if self._allow_mock_fallback:
                logger.warning("Falling back to MOCK mode due to connection failure")
                self._mock_mode_active = True
            return False

    def _verify_hardware(self) -> bool:
        try:
            test_register = self._get_brightness_register(list(self.channels.keys())[0])
            result = self._client.read_holding_registers(test_register, 1)

            if result.isError():
                logger.error(f"Hardware verification failed: Modbus read error - {result}")
                return False

            if not hasattr(result, 'registers') or result.registers is None:
                logger.error("Hardware verification failed: Invalid response from controller")
                return False

            logger.debug(
                f"Hardware verification passed. Register {test_register} = {result.registers[0]}"
            )
            return True

        except Exception as e:
            logger.error(f"Hardware verification error: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            for timer in self._strobe_timers.values():
                timer.cancel()
            self._strobe_timers.clear()

            for ch_id in self.channels:
                try:
                    self.set_mode(ch_id, LightMode.OFF)
                except:
                    pass

            if self._client:
                self._client.close()
                self._client = None

            self._is_connected = False
            self._hardware_ready = False
            self._mock_mode_active = False
            logger.info("Light controller disconnected")
            return True

        except Exception as e:
            logger.error(f"Failed to disconnect light controller: {e}")
            return False

    def _check_connection(self) -> bool:
        if not self._is_connected or not self._hardware_ready or not self._client:
            if self._mock_mode_active:
                logger.warning("Operation in MOCK mode - NO hardware control!")
                return False
            logger.error("Light controller not connected")
            return False

        if not self._client.is_socket_open():
            logger.warning("Modbus connection lost, attempting to reconnect...")
            try:
                self._client.connect()
                self._verify_hardware()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                self._is_connected = False
                self._hardware_ready = False
                return False

        return True

    def set_brightness(self, channel_id: str, brightness: int) -> bool:
        if channel_id not in self.channels:
            logger.error(f"Unknown channel: {channel_id}")
            return False

        brightness = max(0, min(100, brightness))

        if not self._check_connection():
            if self._mock_mode_active:
                self._channel_states[channel_id]["brightness"] = brightness
                self.channels[channel_id].brightness = brightness
                logger.debug(f"[MOCK] Channel {channel_id} brightness set to {brightness}%")
                return True
            return False

        try:
            register_addr = self._get_brightness_register(channel_id)
            result = self._client.write_register(register_addr, brightness)

            if result.isError():
                raise RuntimeError(f"Modbus write error: {result}")

            self._channel_states[channel_id]["brightness"] = brightness
            self.channels[channel_id].brightness = brightness
            logger.debug(f"Channel {channel_id} brightness set to {brightness}%")
            return True

        except Exception as e:
            logger.error(f"Failed to set brightness for {channel_id}: {e}")
            self._is_connected = False
            return False

    def set_mode(self, channel_id: str, mode: LightMode) -> bool:
        if channel_id not in self.channels:
            logger.error(f"Unknown channel: {channel_id}")
            return False

        if not self._check_connection():
            if self._mock_mode_active:
                self._channel_states[channel_id]["mode"] = mode
                self._channel_states[channel_id]["is_on"] = mode != LightMode.OFF
                self.channels[channel_id].mode = mode
                logger.debug(f"[MOCK] Channel {channel_id} mode set to {mode.value}")
                return True
            return False

        try:
            register_addr = self._get_mode_register(channel_id)
            mode_value = {"off": 0, "continuous": 1, "strobe": 2}[mode.value]
            result = self._client.write_register(register_addr, mode_value)

            if result.isError():
                raise RuntimeError(f"Modbus write error: {result}")

            self._channel_states[channel_id]["mode"] = mode
            self._channel_states[channel_id]["is_on"] = mode != LightMode.OFF
            self.channels[channel_id].mode = mode
            logger.debug(f"Channel {channel_id} mode set to {mode.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set mode for {channel_id}: {e}")
            self._is_connected = False
            return False

    def set_color_temp(self, channel_id: str, color_temp: int) -> bool:
        if channel_id not in self.channels:
            logger.error(f"Unknown channel: {channel_id}")
            return False

        color_temp = max(2700, min(6500, color_temp))

        if not self._check_connection():
            if self._mock_mode_active:
                self._channel_states[channel_id]["color_temp"] = color_temp
                self.channels[channel_id].color_temp = color_temp
                logger.debug(f"[MOCK] Channel {channel_id} color temp set to {color_temp}K")
                return True
            return False

        try:
            register_addr = self._get_color_temp_register(channel_id)
            result = self._client.write_register(register_addr, color_temp)

            if result.isError():
                raise RuntimeError(f"Modbus write error: {result}")

            self._channel_states[channel_id]["color_temp"] = color_temp
            self.channels[channel_id].color_temp = color_temp
            logger.debug(f"Channel {channel_id} color temp set to {color_temp}K")
            return True

        except Exception as e:
            logger.error(f"Failed to set color temp for {channel_id}: {e}")
            self._is_connected = False
            return False

    def trigger_strobe(self, channel_id: str, delay_us: int = 0, width_us: int = 1000) -> bool:
        if channel_id not in self.channels:
            logger.error(f"Unknown channel: {channel_id}")
            return False

        state = self._channel_states[channel_id]
        if state["mode"] != LightMode.STROBE:
            return True

        if not self._check_connection():
            if self._mock_mode_active:
                state["is_on"] = True
                logger.debug(f"[MOCK] Strobe triggered on channel {channel_id}, width: {width_us}us")

                def _turn_off():
                    state["is_on"] = False

                timer = threading.Timer(width_us / 1_000_000.0, _turn_off)
                timer.start()
                return True
            return False

        try:
            if channel_id in self._strobe_timers:
                self._strobe_timers[channel_id].cancel()

            trigger_addr = self._get_strobe_register(channel_id)
            result = self._client.write_register(trigger_addr, 1)

            if result.isError():
                raise RuntimeError(f"Modbus write error: {result}")

            state["is_on"] = True
            logger.debug(f"Strobe triggered on channel {channel_id}, width: {width_us}us")

            def _turn_off():
                try:
                    state["is_on"] = False
                    if self._check_connection():
                        self._client.write_register(trigger_addr, 0)
                except:
                    pass

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
        state = self._channel_states.get(channel_id)
        if state:
            return {**state, "mock_mode": self._mock_mode_active}
        return None

    @property
    def is_mock_mode(self) -> bool:
        return self._mock_mode_active

    @property
    def hardware_ready(self) -> bool:
        return self._hardware_ready and self._is_connected
