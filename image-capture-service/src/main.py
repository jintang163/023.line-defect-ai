import os
import sys
import time
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import ConfigManager
from src.utils.logger import Logger
from src.capture.camera_manager import CameraManager
from src.capture.trigger_controller import HardwareTriggerController as TriggerController
from src.lighting.light_controller import ModbusLightController
from src.preprocessing.image_processor import ImagePreprocessor
from src.buffer.ring_buffer import RingBuffer
from src.buffer.local_cache import LocalCache
from src.messaging.message_sender import MessageSender
from src.monitoring.camera_monitor import CameraMonitor

logger = Logger("capture-service", "INFO", "./logs/capture-service.log").logger


class CaptureService:
    def __init__(self, config_path: str = "./config/config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self._is_running = False
        self._shutdown_event = threading.Event()

        self._init_components()
        self._init_callbacks()

    def _init_components(self):
        logger.info("Initializing capture service components...")

        buf_cfg = self.config_manager.get_buffer_config()
        self.ring_buffer = RingBuffer(buf_cfg.get("ring_buffer_size", 100))
        self.local_cache = LocalCache(
            cache_dir=buf_cfg.get("local_cache_dir", "./data/cache"),
            max_size_gb=buf_cfg.get("max_cache_size_gb", 10),
            retry_interval=buf_cfg.get("retry_interval", 30),
            max_retry=buf_cfg.get("max_retry", 10)
        )

        self.camera_manager = CameraManager(self.config_manager)

        light_cfg = self.config_manager.get_lighting_config()
        self.light_controller = ModbusLightController(
            channels=self.config_manager.get_light_channels(),
            host=light_cfg.get("host", "192.168.1.200"),
            port=light_cfg.get("port", 502),
            slave_id=light_cfg.get("slave_id", 1),
            presets=light_cfg.get("presets", {}),
            allow_mock_fallback=light_cfg.get("allow_mock_fallback", False)
        )

        self.image_preprocessor = ImagePreprocessor(self.config_manager)
        self.message_sender = MessageSender(self.config_manager, self.ring_buffer, self.local_cache)
        self.camera_monitor = CameraMonitor(self.config_manager, self.camera_manager)
        self.trigger_controller = TriggerController(self.config_manager)

        logger.info("All components initialized")

    def _init_callbacks(self):
        self.trigger_controller.set_callback(self._on_trigger)
        self.camera_manager.add_trigger_callback(self._on_capture_complete)

    def start(self):
        if self._is_running:
            logger.warning("Service already running")
            return

        logger.info("Starting capture service...")

        if not self.camera_manager.initialize():
            logger.error("Failed to initialize cameras")
            return

        light_connected = self.light_controller.connect()
        if not light_connected:
            logger.warning("Light controller not connected - check hardware connection")

        self.message_sender.start()
        self.camera_monitor.start()
        self.trigger_controller.start()

        self._start_http_server()

        self._is_running = True
        logger.info("Capture service started successfully")

    def stop(self):
        if not self._is_running:
            return

        logger.info("Stopping capture service...")
        self._shutdown_event.set()

        self.trigger_controller.stop()
        self.camera_monitor.stop()
        self.message_sender.stop()
        self.light_controller.disconnect()
        self.camera_manager.disconnect_all()

        if hasattr(self, '_http_server'):
            self._http_server.shutdown()

        self._is_running = False
        logger.info("Capture service stopped")

    def _on_trigger(self):
        logger.debug("Trigger received, initiating capture...")

        self.light_controller.trigger_all_strobe()

        message = self.camera_manager.trigger_sync_capture()
        if message:
            for img in message.images:
                self.image_preprocessor.process(img)

    def _on_capture_complete(self, message):
        logger.debug(f"Capture complete for sequence {message.sequence_id}")
        self.message_sender.enqueue_message(message)

    def _start_http_server(self):
        service = self
        cfg = self.config_manager.get_service_config()
        http_port = cfg.get("http_port", 8000)

        class APIHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                logger.debug(f"HTTP {args[0]} {args[1]}")

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                try:
                    if path == "/health":
                        self._send_json(200, {"status": "ok", "running": service._is_running})

                    elif path == "/api/status":
                        self._send_json(200, service._get_status())

                    elif path == "/api/cameras":
                        self._send_json(200, {"cameras": service._get_cameras_status()})

                    elif path == "/api/alerts":
                        limit = int(params.get("limit", [100])[0])
                        alerts = service.camera_monitor._alert_manager.get_recent_alerts(limit)
                        self._send_json(200, {"alerts": [a.__dict__ for a in alerts]})

                    elif path == "/api/stats":
                        self._send_json(200, service.message_sender.get_stats())

                    elif path == "/api/trigger":
                        service.trigger_controller.manual_trigger()
                        self._send_json(200, {"status": "triggered"})

                    elif path == "/api/config":
                        self._send_json(200, service.config_manager.config)

                    elif path == "/api/reload-config":
                        service.config_manager.reload()
                        service.image_preprocessor.reload_config()
                        self._send_json(200, {"status": "reloaded"})

                    else:
                        self._send_json(404, {"error": "Not found"})

                except Exception as e:
                    logger.error(f"API error: {e}")
                    self._send_json(500, {"error": str(e)})

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path

                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body) if body else {}

                    if path == "/api/camera/exposure":
                        camera_id = data.get("camera_id")
                        exposure = data.get("exposure_time")
                        if camera_id and exposure:
                            success = service.camera_manager.set_camera_exposure(camera_id, int(exposure))
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing camera_id or exposure_time"})

                    elif path == "/api/camera/gain":
                        camera_id = data.get("camera_id")
                        gain = data.get("gain")
                        if camera_id and gain is not None:
                            success = service.camera_manager.set_camera_gain(camera_id, float(gain))
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing camera_id or gain"})

                    elif path == "/api/cameras/exposure":
                        exposure = data.get("exposure_time")
                        if exposure:
                            results = service.camera_manager.set_all_exposure(int(exposure))
                            self._send_json(200, {"results": results})
                        else:
                            self._send_json(400, {"error": "Missing exposure_time"})

                    elif path == "/api/cameras/gain":
                        gain = data.get("gain")
                        if gain is not None:
                            results = service.camera_manager.set_all_gain(float(gain))
                            self._send_json(200, {"results": results})
                        else:
                            self._send_json(400, {"error": "Missing gain"})

                    elif path == "/api/light/brightness":
                        channel_id = data.get("channel_id")
                        brightness = data.get("brightness")
                        if channel_id and brightness is not None:
                            success = service.light_controller.set_brightness(channel_id, int(brightness))
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing channel_id or brightness"})

                    elif path == "/api/light/mode":
                        channel_id = data.get("channel_id")
                        mode = data.get("mode")
                        from src.utils.schemas import LightMode
                        if channel_id and mode:
                            success = service.light_controller.set_mode(channel_id, LightMode(mode))
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing channel_id or mode"})

                    elif path == "/api/light/color_temp":
                        channel_id = data.get("channel_id")
                        color_temp = data.get("color_temp")
                        if channel_id and color_temp:
                            success = service.light_controller.set_color_temp(channel_id, int(color_temp))
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing channel_id or color_temp"})

                    elif path == "/api/light/preset":
                        material = data.get("material")
                        if material:
                            success = service.light_controller.apply_material_preset(material)
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing material"})

                    elif path == "/api/preprocessing/config":
                        service.image_preprocessor.update_config(data)
                        self._send_json(200, {"status": "updated"})

                    elif path == "/api/trigger/interval":
                        interval = data.get("interval")
                        if interval:
                            service.trigger_controller.set_simulation_interval(float(interval))
                            self._send_json(200, {"status": "updated"})
                        else:
                            self._send_json(400, {"error": "Missing interval"})

                    elif path == "/api/mq/reconnect":
                        success = service.message_sender.reconnect()
                        self._send_json(200, {"success": success})

                    elif path == "/api/storage/reconnect":
                        success = service.message_sender.reconnect_storage()
                        self._send_json(200, {"success": success})

                    else:
                        self._send_json(404, {"error": "Not found"})

                except json.JSONDecodeError:
                    self._send_json(400, {"error": "Invalid JSON"})
                except Exception as e:
                    logger.error(f"POST API error: {e}")
                    self._send_json(500, {"error": str(e)})

            def _send_json(self, status_code: int, data: Dict[str, Any]):
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))

        def _run_server():
            try:
                service._http_server = HTTPServer(("0.0.0.0", http_port), APIHandler)
                logger.info(f"HTTP API server started on port {http_port}")
                service._http_server.serve_forever()
            except Exception as e:
                logger.error(f"HTTP server error: {e}")

        threading.Thread(target=_run_server, daemon=True).start()

    def _get_status(self) -> Dict[str, Any]:
        return {
            "running": self._is_running,
            "timestamp": time.time(),
            "trigger_count": self.trigger_controller.trigger_count,
            "camera_count": len(self.camera_manager.cameras),
            "mq_connected": self.message_sender._producer.is_connected() if self.message_sender._producer else False,
            "light_connected": self.light_controller.is_connected(),
            "light_hardware_ready": self.light_controller.hardware_ready if hasattr(self.light_controller, 'hardware_ready') else False,
            "light_mock_mode": self.light_controller.is_mock_mode if hasattr(self.light_controller, 'is_mock_mode') else False,
            "trigger_mode": "hardware" if hasattr(self.trigger_controller, 'simulation_mode') and not self.trigger_controller.simulation_mode else "simulation",
            "ptp_enabled": self.trigger_controller.enable_ptp if hasattr(self.trigger_controller, 'enable_ptp') else False,
            "cameras": self._get_cameras_status(),
            "lights": [
                {
                    "id": ch.id,
                    "name": ch.name,
                    "mode": ch.mode.value,
                    "brightness": ch.brightness,
                    "color_temp": ch.color_temp
                }
                for ch in self.light_controller.get_all_channels()
            ]
        }

    def _get_cameras_status(self):
        return [
            {
                "camera_id": s.camera_id,
                "status": s.status.value,
                "exposure_time": s.exposure_time,
                "gain": s.gain,
                "trigger_count": s.trigger_count,
                "last_capture_time": s.last_capture_time,
                "temperature": s.temperature,
                "error_message": s.error_message
            }
            for s in self.camera_manager.get_camera_statuses()
        ]

    def wait(self):
        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.stop()


def main():
    config_path = os.environ.get("CONFIG_PATH", "./config/config.yaml")
    service = CaptureService(config_path)

    try:
        service.start()
        service.wait()
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        service.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
