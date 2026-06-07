import threading
import time
import json
from typing import List, Dict, Any, Optional, Callable
from prometheus_client import Gauge, Counter, Info, start_http_server
import psutil
from src.capture.camera_manager import CameraManager
from src.config.settings import ConfigManager
from src.utils.schemas import AlertMessage, CameraStatus, CameraStatusInfo
from src.utils.logger import Logger

logger = Logger().logger


class CameraMonitor:
    def __init__(self, config_manager: ConfigManager, camera_manager: CameraManager):
        self.config_manager = config_manager
        self.camera_manager = camera_manager
        self._monitor_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._alert_callbacks: List[Callable[[AlertMessage], None]] = []
        self._last_statuses: Dict[str, CameraStatusInfo] = {}

        mon_cfg = config_manager.get_monitoring_config()
        self.status_interval = mon_cfg.get("status_report_interval", 5)
        self.health_interval = mon_cfg.get("health_check_interval", 10)
        self.enable_prometheus = mon_cfg.get("enable_prometheus", True)
        self.prometheus_port = mon_cfg.get("prometheus_port", 9090)

        self._init_prometheus_metrics()
        self._alert_manager = AlertManager(config_manager)

    def _init_prometheus_metrics(self):
        self.metric_camera_status = Gauge(
            "defect_camera_status", "Camera status (0=offline, 1=online, 2=error)",
            ["camera_id", "camera_position"]
        )
        self.metric_camera_exposure = Gauge(
            "defect_camera_exposure_time_us", "Camera exposure time in microseconds",
            ["camera_id"]
        )
        self.metric_camera_gain = Gauge(
            "defect_camera_gain", "Camera gain",
            ["camera_id"]
        )
        self.metric_trigger_count = Counter(
            "defect_camera_trigger_count_total", "Total trigger count",
            ["camera_id"]
        )
        self.metric_capture_count = Counter(
            "defect_capture_count_total", "Total capture count"
        )
        self.metric_capture_latency = Gauge(
            "defect_capture_latency_seconds", "Capture latency in seconds"
        )
        self.metric_system_cpu = Gauge(
            "defect_system_cpu_percent", "System CPU usage percent"
        )
        self.metric_system_memory = Gauge(
            "defect_system_memory_percent", "System memory usage percent"
        )
        self.metric_service_info = Info(
            "defect_capture_service", "Capture service information"
        )

        service_cfg = self.config_manager.get_service_config()
        self.metric_service_info.info({
            "name": service_cfg.get("name", "capture-service"),
            "instance_id": service_cfg.get("instance_id", "capture-001"),
            "version": "1.0.0"
        })

    def add_alert_callback(self, callback: Callable[[AlertMessage], None]):
        self._alert_callbacks.append(callback)

    def start(self):
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")

        self._alert_manager.start()

        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._status_report_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Camera status monitor started")

        if self._health_thread is None or not self._health_thread.is_alive():
            self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._health_thread.start()
            logger.info("Health check thread started")

    def stop(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self._health_thread:
            self._health_thread.join(timeout=5)
        self._alert_manager.stop()
        logger.info("Camera monitor stopped")

    def _status_report_loop(self):
        while not self._stop_event.is_set():
            try:
                statuses = self.camera_manager.get_camera_statuses()
                self._update_metrics(statuses)
                self._check_status_changes(statuses)
                self._update_system_metrics()
            except Exception as e:
                logger.error(f"Status report error: {e}")

            self._stop_event.wait(self.status_interval)

    def _health_check_loop(self):
        while not self._stop_event.is_set():
            try:
                health = self.camera_manager.health_check()
                for cam_id, is_healthy in health.items():
                    if not is_healthy:
                        alert = AlertMessage.create(
                            level="critical",
                            category="camera_health",
                            message=f"Camera {cam_id} is not responding",
                            source="camera_monitor",
                            details={"camera_id": cam_id}
                        )
                        self._send_alert(alert)
                        self._try_reconnect_camera(cam_id)
            except Exception as e:
                logger.error(f"Health check error: {e}")

            self._stop_event.wait(self.health_interval)

    def _update_metrics(self, statuses: List[CameraStatusInfo]):
        status_value_map = {
            CameraStatus.OFFLINE: 0,
            CameraStatus.ONLINE: 1,
            CameraStatus.INITIALIZING: 1,
            CameraStatus.CAPTURING: 1,
            CameraStatus.ERROR: 2
        }

        for status in statuses:
            camera = self.camera_manager.get_camera(status.camera_id)
            position = camera.config.position if camera else "unknown"

            self.metric_camera_status.labels(
                camera_id=status.camera_id,
                camera_position=position
            ).set(status_value_map.get(status.status, 0))

            self.metric_camera_exposure.labels(
                camera_id=status.camera_id
            ).set(status.exposure_time)

            self.metric_camera_gain.labels(
                camera_id=status.camera_id
            ).set(status.gain)

    def _check_status_changes(self, statuses: List[CameraStatusInfo]):
        for status in statuses:
            prev = self._last_statuses.get(status.camera_id)
            if prev and prev.status != status.status:
                if status.status == CameraStatus.ERROR:
                    alert = AlertMessage.create(
                        level="error",
                        category="camera_error",
                        message=f"Camera {status.camera_id} error: {status.error_message}",
                        source="camera_monitor",
                        details={
                            "camera_id": status.camera_id,
                            "error": status.error_message,
                            "previous_status": prev.status.value
                        }
                    )
                    self._send_alert(alert)

                elif status.status == CameraStatus.OFFLINE and prev.status != CameraStatus.OFFLINE:
                    alert = AlertMessage.create(
                        level="warning",
                        category="camera_offline",
                        message=f"Camera {status.camera_id} went offline",
                        source="camera_monitor",
                        details={"camera_id": status.camera_id}
                    )
                    self._send_alert(alert)

                elif status.status == CameraStatus.ONLINE and prev.status == CameraStatus.OFFLINE:
                    alert = AlertMessage.create(
                        level="info",
                        category="camera_online",
                        message=f"Camera {status.camera_id} came online",
                        source="camera_monitor",
                        details={"camera_id": status.camera_id}
                    )
                    self._send_alert(alert)

            self._last_statuses[status.camera_id] = status

    def _update_system_metrics(self):
        self.metric_system_cpu.set(psutil.cpu_percent())
        self.metric_system_memory.set(psutil.virtual_memory().percent)

    def _try_reconnect_camera(self, camera_id: str):
        logger.info(f"Attempting to reconnect camera {camera_id}")
        self.camera_manager.reconnect_camera(camera_id)

    def _send_alert(self, alert: AlertMessage):
        self._alert_manager.send_alert(alert)
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        return {
            "cameras": [s.__dict__ for s in self.camera_manager.get_camera_statuses()],
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "trigger_count": self.camera_manager.trigger_count
            },
            "timestamp": time.time()
        }


class AlertManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._alerts: List[AlertMessage] = []
        self._max_alerts = 1000
        self._lock = threading.Lock()

        mon_cfg = config_manager.get_monitoring_config()
        self.webhook_url = mon_cfg.get("alert_webhook")
        self.enable_email = mon_cfg.get("enable_email_alert", False)
        self.smtp_cfg = mon_cfg.get("smtp", {})

    def start(self):
        pass

    def stop(self):
        pass

    def send_alert(self, alert: AlertMessage):
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > self._max_alerts:
                self._alerts.pop(0)

        logger.warning(f"ALERT [{alert.level}] {alert.category}: {alert.message}")

        if self.webhook_url:
            self._send_webhook_alert(alert)

        if self.enable_email:
            self._send_email_alert(alert)

    def _send_webhook_alert(self, alert: AlertMessage):
        try:
            import urllib.request
            data = json.dumps({
                "alert_id": alert.alert_id,
                "level": alert.level,
                "category": alert.category,
                "message": alert.message,
                "source": alert.source,
                "timestamp": alert.timestamp,
                "details": alert.details
            }).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            def _send():
                try:
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        resp.read()
                except Exception as e:
                    logger.error(f"Webhook alert failed: {e}")

            threading.Thread(target=_send, daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to create webhook alert: {e}")

    def _send_email_alert(self, alert: AlertMessage):
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            subject = f"[{alert.level.upper()}] {alert.category}: {alert.message[:50]}"
            body = f"""
            Alert ID: {alert.alert_id}
            Level: {alert.level}
            Category: {alert.category}
            Message: {alert.message}
            Source: {alert.source}
            Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
            Details: {json.dumps(alert.details, indent=2)}
            """

            msg = MIMEMultipart()
            msg["From"] = self.smtp_cfg.get("username")
            msg["To"] = ", ".join(self.smtp_cfg.get("recipients", []))
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            def _send():
                try:
                    with smtplib.SMTP(self.smtp_cfg.get("host"), self.smtp_cfg.get("port")) as server:
                        server.starttls()
                        server.login(self.smtp_cfg.get("username"), self.smtp_cfg.get("password"))
                        server.send_message(msg)
                except Exception as e:
                    logger.error(f"Email alert failed: {e}")

            threading.Thread(target=_send, daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to create email alert: {e}")

    def get_recent_alerts(self, limit: int = 100) -> List[AlertMessage]:
        with self._lock:
            return self._alerts[-limit:]
