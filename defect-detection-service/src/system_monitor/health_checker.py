import subprocess
import shutil
import platform
import threading
import time
from typing import Dict, List, Optional

from src.utils.logger import Logger

logger = Logger("health_checker", "INFO", "./logs/defect-detection.log").logger


class HealthChecker:
    def __init__(self, config: dict, message_consumer=None, result_producer=None, algorithm_manager=None):
        self._enable = config.get("enable", False)
        self._check_interval_sec = config.get("check_interval_sec", 30)
        self._auto_restart = config.get("auto_restart", True)
        self._restart_cooldown_sec = config.get("restart_cooldown_sec", 60)
        self._disk_threshold_percent = config.get("disk_threshold_percent", 90)
        self._gpu_threshold_percent = config.get("gpu_threshold_percent", 95)
        self._mq_backlog_threshold = config.get("mq_backlog_threshold", 1000)
        self._camera_timeout_sec = config.get("camera_timeout_sec", 10)

        self._message_consumer = message_consumer
        self._result_producer = result_producer
        self._algorithm_manager = algorithm_manager

        self._health_status: Dict[str, Dict] = {
            "camera": {"status": "ok", "message": "", "last_check": 0.0, "details": {}},
            "gpu": {"status": "ok", "message": "", "last_check": 0.0, "details": {}},
            "mq": {"status": "ok", "message": "", "last_check": 0.0, "details": {}},
            "disk": {"status": "ok", "message": "", "last_check": 0.0, "details": {}},
        }

        self._check_history: List[Dict] = []
        self._max_history = 500

        self._stop_event = threading.Event()
        self._check_thread: Optional[threading.Thread] = None
        self._last_restart_times: Dict[str, float] = {}

    def start(self):
        if not self._enable:
            logger.info("Health checker is disabled")
            return
        if self._check_thread is not None and self._check_thread.is_alive():
            logger.warning("Health checker is already running")
            return
        self._stop_event.clear()
        self._check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self._check_thread.start()
        logger.info("Health checker started with interval %d sec", self._check_interval_sec)

    def stop(self):
        self._stop_event.set()
        if self._check_thread is not None:
            self._check_thread.join(timeout=5)
            self._check_thread = None
        logger.info("Health checker stopped")

    def _check_loop(self):
        while not self._stop_event.is_set():
            try:
                self.check_all()
            except Exception as e:
                logger.error("Health check loop error: %s", str(e))
            self._stop_event.wait(self._check_interval_sec)

    def check_all(self) -> Dict:
        self.check_camera()
        self.check_gpu()
        self.check_mq()
        self.check_disk()
        self._record_history()
        return self._health_status.copy()

    def check_camera(self) -> Dict:
        now = time.time()
        details = {}
        status = "ok"
        message = "Camera status is ok"

        if self._message_consumer is not None:
            try:
                connected = getattr(self._message_consumer, "is_connected", None)
                if callable(connected):
                    if connected():
                        status = "ok"
                        message = "Camera connection is active"
                    else:
                        status = "error"
                        message = "Camera connection is lost"
                else:
                    status = "ok"
                    message = "Camera consumer available, connectivity check not supported"
            except Exception as e:
                status = "warning"
                message = f"Camera check failed: {str(e)}"
        else:
            status = "ok"
            message = "Camera check skipped (no consumer reference)"

        self._health_status["camera"] = {
            "status": status,
            "message": message,
            "last_check": now,
            "details": details,
        }
        return self._health_status["camera"].copy()

    def check_gpu(self) -> Dict:
        now = time.time()
        details = {}
        status = "ok"
        message = "GPU status is ok"

        gpu_info = self._get_nvidia_smi_info()
        if gpu_info is not None:
            details = gpu_info
            utilization = gpu_info.get("utilization_percent", 0)
            memory_used_percent = gpu_info.get("memory_used_percent", 0)
            temperature = gpu_info.get("temperature_c", 0)

            if utilization >= self._gpu_threshold_percent or memory_used_percent >= self._gpu_threshold_percent:
                status = "warning"
                message = f"GPU usage high: util={utilization}%, mem={memory_used_percent}%"
            elif temperature >= 90:
                status = "warning"
                message = f"GPU temperature high: {temperature}C"
            else:
                status = "ok"
                message = f"GPU normal: util={utilization}%, mem={memory_used_percent}%, temp={temperature}C"
        else:
            gputil_info = self._get_gputil_info()
            if gputil_info is not None:
                details = gputil_info
                status = "ok"
                message = "GPU info retrieved via GPUtil"
            else:
                status = "unknown"
                message = "GPU status unknown (nvidia-smi and GPUtil unavailable)"

        self._health_status["gpu"] = {
            "status": status,
            "message": message,
            "last_check": now,
            "details": details,
        }
        return self._health_status["gpu"].copy()

    def _get_nvidia_smi_info(self) -> Optional[Dict]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                utilization = float(parts[0])
                memory_used = float(parts[1])
                memory_total = float(parts[2])
                temperature = float(parts[3])
                memory_used_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                return {
                    "utilization_percent": utilization,
                    "memory_used_mb": memory_used,
                    "memory_total_mb": memory_total,
                    "memory_used_percent": round(memory_used_percent, 1),
                    "temperature_c": temperature,
                }
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            return None

    def _get_gputil_info(self) -> Optional[Dict]:
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            gpu = gpus[0]
            return {
                "utilization_percent": gpu.load * 100,
                "memory_used_mb": gpu.memoryUsed,
                "memory_total_mb": gpu.memoryTotal,
                "memory_used_percent": (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                "temperature_c": gpu.temperature,
            }
        except Exception:
            return None

    def check_mq(self) -> Dict:
        now = time.time()
        details = {}
        status = "ok"
        message = "MQ status is ok"

        consumer_ok = True
        producer_ok = True

        if self._message_consumer is not None:
            try:
                connected = getattr(self._message_consumer, "is_connected", None)
                if callable(connected):
                    consumer_ok = connected()
            except Exception:
                consumer_ok = False

        if self._result_producer is not None:
            try:
                connected = getattr(self._result_producer, "is_connected", None)
                if callable(connected):
                    producer_ok = connected()
            except Exception:
                producer_ok = False

        details["consumer_connected"] = consumer_ok
        details["producer_connected"] = producer_ok

        backlog = self._check_mq_backlog()
        if backlog is not None:
            details["backlog"] = backlog
            if backlog > self._mq_backlog_threshold:
                status = "warning"
                message = f"MQ backlog high: {backlog} (threshold: {self._mq_backlog_threshold})"
            elif not consumer_ok or not producer_ok:
                status = "error"
                message = f"MQ connection lost: consumer={consumer_ok}, producer={producer_ok}"
            else:
                status = "ok"
                message = f"MQ normal: backlog={backlog}"
        else:
            if not consumer_ok or not producer_ok:
                status = "error"
                message = f"MQ connection lost: consumer={consumer_ok}, producer={producer_ok}"
            else:
                status = "ok"
                message = "MQ connections ok, backlog unavailable"

        self._health_status["mq"] = {
            "status": status,
            "message": message,
            "last_check": now,
            "details": details,
        }

        if status == "error" and self._auto_restart:
            self._auto_restart_service("mq")

        return self._health_status["mq"].copy()

    def _check_mq_backlog(self) -> Optional[int]:
        try:
            import requests

            host = "localhost"
            port = 15672
            username = "guest"
            password = "guest"

            if self._message_consumer is not None:
                host = getattr(self._message_consumer, "mq_host", host)
                port = getattr(self._message_consumer, "mq_management_port", port)
                username = getattr(self._message_consumer, "mq_user", username)
                password = getattr(self._message_consumer, "mq_password", password)

            url = f"http://{host}:{port}/api/queues"
            resp = requests.get(url, auth=(username, password), timeout=5)
            if resp.status_code == 200:
                queues = resp.json()
                total_messages = sum(q.get("messages", 0) for q in queues)
                return total_messages
        except Exception:
            pass
        return None

    def check_disk(self) -> Dict:
        now = time.time()
        details = {}
        status = "ok"
        message = "Disk status is ok"

        paths_to_check = ["./data", "./logs", "."]
        for path in paths_to_check:
            try:
                usage = shutil.disk_usage(path)
                used_percent = usage.used / usage.total * 100
                details[path] = {
                    "total_gb": round(usage.total / (1024 ** 3), 2),
                    "used_gb": round(usage.used / (1024 ** 3), 2),
                    "free_gb": round(usage.free / (1024 ** 3), 2),
                    "used_percent": round(used_percent, 1),
                }
                if used_percent >= self._disk_threshold_percent:
                    status = "warning"
                    message = f"Disk usage high on {path}: {used_percent:.1f}%"
            except Exception as e:
                details[path] = {"error": str(e)}

        self._health_status["disk"] = {
            "status": status,
            "message": message,
            "last_check": now,
            "details": details,
        }
        return self._health_status["disk"].copy()

    def _auto_restart_service(self, service_name: str):
        if not self._auto_restart:
            logger.info("Auto-restart is disabled, skipping restart for %s", service_name)
            return

        now = time.time()
        last_restart = self._last_restart_times.get(service_name, 0)
        if now - last_restart < self._restart_cooldown_sec:
            logger.info(
                "Restart cooldown not elapsed for %s (%.0f sec remaining)",
                service_name,
                self._restart_cooldown_sec - (now - last_restart),
            )
            return

        logger.warning("Attempting auto-restart for service: %s", service_name)
        self._last_restart_times[service_name] = now

        try:
            if service_name == "mq":
                if self._message_consumer is not None:
                    reconnect = getattr(self._message_consumer, "reconnect", None)
                    if callable(reconnect):
                        reconnect()
                        logger.info("Message consumer reconnected successfully")
                if self._result_producer is not None:
                    reconnect = getattr(self._result_producer, "reconnect", None)
                    if callable(reconnect):
                        reconnect()
                        logger.info("Result producer reconnected successfully")
            elif service_name == "algorithm":
                if self._algorithm_manager is not None:
                    restart = getattr(self._algorithm_manager, "restart", None)
                    if callable(restart):
                        restart()
                        logger.info("Algorithm manager restarted successfully")
            else:
                logger.warning("Unknown service for auto-restart: %s", service_name)
        except Exception as e:
            logger.error("Auto-restart failed for %s: %s", service_name, str(e))

    def _record_history(self):
        record = {
            "timestamp": time.time(),
            "status": {k: v.copy() for k, v in self._health_status.items()},
        }
        self._check_history.append(record)
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history:]

    def get_status(self) -> Dict:
        return self._health_status.copy()

    def get_history(self, limit: int = 100) -> List[Dict]:
        return self._check_history[-limit:]
