import threading
import time
import struct
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass
from src.config.settings import ConfigManager
from src.utils.logger import Logger

logger = Logger().logger


@dataclass
class TriggerSignal:
    timestamp_ns: int
    ptp_time_ns: Optional[int]
    sequence_number: int
    source: str


class HardwareTriggerController:
    def __init__(self, config_manager: ConfigManager,
                 on_trigger_callback: Optional[Callable[[TriggerSignal], None]] = None):
        self.config_manager = config_manager
        self._callback = on_trigger_callback
        self._trigger_thread: Optional[threading.Thread] = None
        self._ptp_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._trigger_count = 0
        self._last_trigger_time = 0.0
        self._hardware_ready = False
        self._ptp_offset_ns = 0
        self._trigger_sequence = 0
        self._sync_lock = threading.Lock()
        self._camera_trigger_signals: Dict[str, threading.Event] = {}

        trig_cfg = config_manager.get_trigger_config()
        self.gpio_pin = trig_cfg.get("sensor_gpio_pin", 17)
        self.debounce_ms = trig_cfg.get("debounce_ms", 50)
        self.simulation_mode = trig_cfg.get("simulation_mode", False)
        self.simulation_interval = trig_cfg.get("simulation_interval", 2.0)
        self.enable_ptp = trig_cfg.get("enable_ptp", True)
        self.ptp_interface = trig_cfg.get("ptp_interface", "eth0")
        self.hardware_trigger_mode = trig_cfg.get("hardware_trigger_mode", "parallel")
        self.trigger_output_pins = trig_cfg.get("trigger_output_pins", [18, 19, 20])

        self._gpio_available = self._try_import_gpio()
        self._ptp_available = self._try_import_ptp()

        if not self._gpio_available and not self.simulation_mode:
            logger.error(
                "RPi.GPIO not available and simulation_mode is disabled. "
                "Trigger controller cannot operate. "
                "Install RPi.GPIO or enable simulation_mode in config."
            )
            raise RuntimeError("GPIO hardware not available and simulation disabled")

    def _try_import_gpio(self) -> bool:
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            return True
        except ImportError:
            logger.warning("RPi.GPIO not installed. Hardware trigger will not be available.")
            return False

    def _try_import_ptp(self) -> bool:
        try:
            import socket
            self._socket = socket
            return True
        except ImportError:
            logger.warning("Socket module not available, PTP time sync disabled")
            return False

    def set_callback(self, callback: Callable[[TriggerSignal], None]):
        self._callback = callback

    def register_camera_trigger(self, camera_id: str) -> threading.Event:
        event = threading.Event()
        self._camera_trigger_signals[camera_id] = event
        return event

    def start(self) -> bool:
        if self._trigger_thread is None or not self._trigger_thread.is_alive():
            self._stop_event.clear()

            if not self._initialize_hardware():
                logger.error("Failed to initialize trigger hardware")
                if not self.simulation_mode:
                    return False

            if self.enable_ptp and self._ptp_available:
                self._ptp_thread = threading.Thread(target=self._ptp_sync_loop, daemon=True)
                self._ptp_thread.start()
                logger.info("PTP time synchronization thread started")

            if self.simulation_mode:
                self._trigger_thread = threading.Thread(target=self._simulation_loop, daemon=True)
                logger.warning(
                    f"Starting trigger in SIMULATION mode, interval: {self.simulation_interval}s. "
                    f"This is for DEVELOPMENT ONLY - not for production use."
                )
            else:
                self._trigger_thread = threading.Thread(target=self._hardware_trigger_loop, daemon=True)
                logger.info(
                    f"Starting trigger in HARDWARE mode, GPIO pin: {self.gpio_pin}, "
                    f"debounce: {self.debounce_ms}ms, mode: {self.hardware_trigger_mode}"
                )

            self._trigger_thread.start()
            self._hardware_ready = True
            return True
        return False

    def stop(self):
        self._stop_event.set()
        self._hardware_ready = False

        if self._trigger_thread:
            self._trigger_thread.join(timeout=5)
        if self._ptp_thread:
            self._ptp_thread.join(timeout=5)

        if self._gpio_available:
            try:
                self._GPIO.cleanup()
            except Exception as e:
                logger.warning(f"GPIO cleanup error: {e}")

        logger.info("Trigger controller stopped")

    def _initialize_hardware(self) -> bool:
        if not self._gpio_available:
            return False

        try:
            self._GPIO.setmode(self._GPIO.BCM)
            self._GPIO.setwarnings(False)

            self._GPIO.setup(self.gpio_pin, self._GPIO.IN, pull_up_down=self._GPIO.PUD_UP)
            self._GPIO.add_event_detect(
                self.gpio_pin,
                self._GPIO.FALLING,
                callback=self._gpio_interrupt_callback,
                bouncetime=self.debounce_ms
            )

            if self.hardware_trigger_mode == "parallel" and self.trigger_output_pins:
                for pin in self.trigger_output_pins:
                    self._GPIO.setup(pin, self._GPIO.OUT, initial=self._GPIO.LOW)
                logger.info(f"Initialized {len(self.trigger_output_pins)} trigger output pins: {self.trigger_output_pins}")

            logger.info(f"Hardware trigger initialized on GPIO pin {self.gpio_pin}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GPIO hardware: {e}")
            return False

    def _gpio_interrupt_callback(self, channel):
        if self._stop_event.is_set():
            return

        try:
            if self._GPIO.input(channel) != self._GPIO.LOW:
                return

            if not self._verify_debounce(channel):
                return

            self._distribute_hardware_trigger()

        except Exception as e:
            logger.error(f"GPIO interrupt error: {e}")

    def _verify_debounce(self, channel) -> bool:
        end_time = time.time() + self.debounce_ms / 1000.0
        while time.time() < end_time:
            if self._GPIO.input(channel) != self._GPIO.LOW:
                return False
            time.sleep(0.001)
        return True

    def _distribute_hardware_trigger(self):
        with self._sync_lock:
            self._trigger_sequence += 1
            seq = self._trigger_sequence

        timestamp_ns = time.time_ns()
        ptp_time = self._get_ptp_time_ns()

        if self.hardware_trigger_mode == "parallel" and self._gpio_available:
            self._pulse_trigger_outputs()

        for event in self._camera_trigger_signals.values():
            event.set()

        signal = TriggerSignal(
            timestamp_ns=timestamp_ns,
            ptp_time_ns=ptp_time,
            sequence_number=seq,
            source="hardware_gpio"
        )

        self._process_trigger_signal(signal)

    def _pulse_trigger_outputs(self, pulse_width_us: int = 10):
        for pin in self.trigger_output_pins:
            self._GPIO.output(pin, self._GPIO.HIGH)

        time.sleep(pulse_width_us / 1_000_000.0)

        for pin in self.trigger_output_pins:
            self._GPIO.output(pin, self._GPIO.LOW)

    def _hardware_trigger_loop(self):
        logger.info("Hardware trigger loop running")
        while not self._stop_event.is_set():
            try:
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Hardware trigger loop error: {e}")
                time.sleep(0.1)

    def _simulation_loop(self):
        logger.warning("Simulation trigger loop running - NOT FOR PRODUCTION")
        while not self._stop_event.is_set():
            try:
                self._simulation_trigger()
            except Exception as e:
                logger.error(f"Simulation trigger error: {e}")

            self._stop_event.wait(self.simulation_interval)

    def _simulation_trigger(self):
        with self._sync_lock:
            self._trigger_sequence += 1
            seq = self._trigger_sequence

        timestamp_ns = time.time_ns()
        ptp_time = self._get_ptp_time_ns()

        for event in self._camera_trigger_signals.values():
            event.set()

        signal = TriggerSignal(
            timestamp_ns=timestamp_ns,
            ptp_time_ns=ptp_time,
            sequence_number=seq,
            source="simulation"
        )

        self._process_trigger_signal(signal)

    def _process_trigger_signal(self, signal: TriggerSignal):
        now = time.time()
        if now - self._last_trigger_time < self.debounce_ms / 1000.0:
            logger.debug(f"Trigger signal #{signal.sequence_number} ignored (debounce)")
            return

        self._trigger_count += 1
        self._last_trigger_time = now

        ptp_info = f", PTP: {signal.ptp_time_ns}ns" if signal.ptp_time_ns else ", PTP: N/A"
        logger.debug(
            f"Trigger #{self._trigger_count} (seq={signal.sequence_number}) "
            f"at {signal.timestamp_ns}ns{ptp_info}, source: {signal.source}"
        )

        if self._callback:
            try:
                self._callback(signal)
            except Exception as e:
                logger.error(f"Trigger callback error: {e}")

    def _get_ptp_time_ns(self) -> Optional[int]:
        if not self.enable_ptp or not self._ptp_available:
            return None

        try:
            return time.time_ns() + self._ptp_offset_ns
        except Exception as e:
            logger.debug(f"PTP time read error: {e}")
            return None

    def _ptp_sync_loop(self):
        logger.info(f"PTP sync thread started, monitoring interface: {self.ptp_interface}")
        while not self._stop_event.is_set():
            try:
                self._sync_ptp_offset()
            except Exception as e:
                logger.debug(f"PTP sync error: {e}")

            self._stop_event.wait(1.0)

    def _sync_ptp_offset(self):
        try:
            import subprocess
            result = subprocess.run(
                ["phc_ctl", self.ptp_interface, "get"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                phc_time = float(result.stdout.strip())
                system_time = time.time()
                self._ptp_offset_ns = int((phc_time - system_time) * 1e9)
                logger.debug(f"PTP offset updated: {self._ptp_offset_ns}ns")
        except Exception as e:
            logger.debug(f"PTP offset sync error: {e}")

    def manual_trigger(self) -> bool:
        if not self._hardware_ready:
            logger.warning("Manual trigger ignored: hardware not ready")
            return False

        logger.info("Manual hardware trigger requested")

        with self._sync_lock:
            self._trigger_sequence += 1
            seq = self._trigger_sequence

        timestamp_ns = time.time_ns()
        ptp_time = self._get_ptp_time_ns()

        if self.hardware_trigger_mode == "parallel" and self._gpio_available:
            self._pulse_trigger_outputs()

        for event in self._camera_trigger_signals.values():
            event.set()

        signal = TriggerSignal(
            timestamp_ns=timestamp_ns,
            ptp_time_ns=ptp_time,
            sequence_number=seq,
            source="manual"
        )

        self._process_trigger_signal(signal)
        return True

    def wait_for_camera_trigger(self, camera_id: str, timeout: float = 1.0) -> bool:
        event = self._camera_trigger_signals.get(camera_id)
        if not event:
            return False

        triggered = event.wait(timeout=timeout)
        if triggered:
            event.clear()
        return triggered

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def is_hardware_ready(self) -> bool:
        return self._hardware_ready

    @property
    def is_simulation_mode(self) -> bool:
        return self.simulation_mode

    def set_simulation_interval(self, interval: float):
        self.simulation_interval = max(0.1, interval)
        logger.info(f"Simulation interval set to {self.simulation_interval}s")

    def set_simulation_mode(self, enabled: bool):
        if enabled and not self._gpio_available:
            logger.warning("Enabling simulation mode because GPIO hardware is not available")
        elif not enabled and not self._gpio_available:
            logger.error("Cannot disable simulation mode: GPIO hardware is not available")
            return

        self.simulation_mode = enabled
        logger.info(f"Simulation mode set to {enabled}")

        if self._trigger_thread and self._trigger_thread.is_alive():
            logger.info("Trigger controller will apply mode change on next restart")

    def get_trigger_stats(self) -> Dict[str, Any]:
        return {
            "trigger_count": self._trigger_count,
            "sequence_number": self._trigger_sequence,
            "hardware_ready": self._hardware_ready,
            "simulation_mode": self.simulation_mode,
            "gpio_available": self._gpio_available,
            "ptp_enabled": self.enable_ptp,
            "ptp_offset_ns": self._ptp_offset_ns,
            "trigger_mode": self.hardware_trigger_mode,
            "last_trigger_time": self._last_trigger_time
        }
