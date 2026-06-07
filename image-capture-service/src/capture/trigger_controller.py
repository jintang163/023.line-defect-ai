import threading
import time
from typing import Optional, Callable
from src.config.settings import ConfigManager
from src.utils.logger import Logger

logger = Logger().logger


class TriggerController:
    def __init__(self, config_manager: ConfigManager, on_trigger_callback: Optional[Callable[[], None]] = None):
        self.config_manager = config_manager
        self._callback = on_trigger_callback
        self._trigger_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._trigger_count = 0
        self._last_trigger_time = 0.0

        trig_cfg = config_manager.get_trigger_config()
        self.gpio_pin = trig_cfg.get("sensor_gpio_pin", 17)
        self.debounce_ms = trig_cfg.get("debounce_ms", 50)
        self.simulation_mode = trig_cfg.get("simulation_mode", True)
        self.simulation_interval = trig_cfg.get("simulation_interval", 2.0)

        self._gpio_available = self._try_import_gpio()

    def _try_import_gpio(self) -> bool:
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            return True
        except ImportError:
            logger.warning("RPi.GPIO not available, will use simulation mode")
            return False

    def set_callback(self, callback: Callable[[], None]):
        self._callback = callback

    def start(self):
        if self._trigger_thread is None or not self._trigger_thread.is_alive():
            self._stop_event.clear()

            if self.simulation_mode or not self._gpio_available:
                self._trigger_thread = threading.Thread(target=self._simulation_loop, daemon=True)
                logger.info(f"Starting trigger in simulation mode, interval: {self.simulation_interval}s")
            else:
                self._trigger_thread = threading.Thread(target=self._gpio_loop, daemon=True)
                logger.info(f"Starting trigger in GPIO mode, pin: {self.gpio_pin}")

            self._trigger_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._trigger_thread:
            self._trigger_thread.join(timeout=5)
        if self._gpio_available:
            try:
                self._GPIO.cleanup()
            except:
                pass
        logger.info("Trigger controller stopped")

    def _simulation_loop(self):
        while not self._stop_event.is_set():
            try:
                self._trigger()
            except Exception as e:
                logger.error(f"Simulation trigger error: {e}")

            self._stop_event.wait(self.simulation_interval)

    def _gpio_loop(self):
        try:
            self._GPIO.setmode(self._GPIO.BCM)
            self._GPIO.setup(self.gpio_pin, self._GPIO.IN, pull_up_down=self._GPIO.PUD_UP)

            while not self._stop_event.is_set():
                try:
                    if self._GPIO.input(self.gpio_pin) == self._GPIO.LOW:
                        if self._check_debounce():
                            self._trigger()
                            while self._GPIO.input(self.gpio_pin) == self._GPIO.LOW:
                                time.sleep(0.001)
                except Exception as e:
                    logger.error(f"GPIO trigger error: {e}")

                time.sleep(0.001)

        except Exception as e:
            logger.error(f"GPIO setup error: {e}")

    def _check_debounce(self) -> bool:
        start = time.time()
        while time.time() - start < self.debounce_ms / 1000.0:
            if self._GPIO.input(self.gpio_pin) != self._GPIO.LOW:
                return False
            time.sleep(0.001)
        return True

    def _trigger(self):
        now = time.time()
        if now - self._last_trigger_time < self.debounce_ms / 1000.0:
            return

        self._trigger_count += 1
        self._last_trigger_time = now
        logger.debug(f"Trigger #{self._trigger_count} at {now}")

        if self._callback:
            try:
                self._callback()
            except Exception as e:
                logger.error(f"Trigger callback error: {e}")

    def manual_trigger(self):
        logger.info("Manual trigger requested")
        self._trigger()

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def set_simulation_interval(self, interval: float):
        self.simulation_interval = max(0.1, interval)
        logger.info(f"Simulation interval set to {self.simulation_interval}s")

    def set_simulation_mode(self, enabled: bool):
        self.simulation_mode = enabled
        logger.info(f"Simulation mode set to {enabled}")
