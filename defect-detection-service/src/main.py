import os
import sys
import time
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import ConfigManager
from src.utils.logger import Logger
from src.utils.schemas import ImageData, DetectionOutput, ProductConfig, AlgorithmType
from src.algorithm_manager import AlgorithmManager
from src.result_annotator import ResultAnnotator
from src.alert_manager import AlertManager
from src.messaging.message_consumer import MessageConsumer
from src.messaging.result_producer import ResultProducer

logger = Logger("defect-detection-service", "INFO", "./logs/defect-detection.log").logger


class DefectDetectionService:
    def __init__(self, config_path: str = "./config/config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self._is_running = False
        self._shutdown_event = threading.Event()

        self._init_components()
        self._init_callbacks()

    def _init_components(self):
        logger.info("Initializing defect detection service components...")

        self.algorithm_manager = AlgorithmManager()

        products_config_path = self.config_manager.get_products_config_path()
        if not self.algorithm_manager.load_products_config(products_config_path):
            logger.warning("Failed to load products configuration")

        self.result_annotator = ResultAnnotator()
        self.alert_manager = AlertManager(
            max_history=1000
        )

        consecutive_threshold = self.config_manager.get_consecutive_ng_threshold()
        self.alert_manager.set_consecutive_ng_threshold(consecutive_threshold)

        msg_config = self.config_manager.get_messaging_config()
        self.message_consumer = MessageConsumer(msg_config)
        self.result_producer = ResultProducer(msg_config)

        logger.info("All components initialized")

    def _init_callbacks(self):
        self.message_consumer.set_callback(self._on_image_received)

    def start(self):
        if self._is_running:
            logger.warning("Service already running")
            return

        logger.info("Starting defect detection service...")

        default_product = os.environ.get("DEFAULT_PRODUCT_ID", None)
        if default_product:
            self.switch_product(default_product)

        self.result_producer.connect()
        self.message_consumer.start()

        self._start_http_server()

        self._is_running = True
        logger.info("Defect detection service started successfully")

    def stop(self):
        if not self._is_running:
            return

        logger.info("Stopping defect detection service...")
        self._shutdown_event.set()

        self.message_consumer.stop()
        self.result_producer.disconnect()
        self.algorithm_manager.cleanup()

        if hasattr(self, '_http_server'):
            self._http_server.shutdown()

        self._is_running = False
        logger.info("Defect detection service stopped")

    def _on_image_received(self, image_data: ImageData):
        try:
            logger.debug(f"Processing image: {image_data.image_id} from {image_data.camera_id}")

            detection_output = self.algorithm_manager.detect(image_data)

            product_config = self.algorithm_manager.get_product_config()
            if product_config:
                alerts = self.alert_manager.process_detection_result(
                    detection_output, product_config
                )
                for alert in alerts:
                    logger.info(f"Alert generated: {alert.level} - {alert.message}")

            annotated = self.result_annotator.annotate(
                image_data.image,
                detection_output.defects,
                product_config
            )
            detection_output.annotated_image = annotated

            self.result_producer.send_result(detection_output, annotated)

            result_icon = "✓" if detection_output.result.value == "OK" else "✗"
            logger.info(
                f"Detection complete: {result_icon} {detection_output.result.value} | "
                f"Defects: {len(detection_output.defects)} | "
                f"Time: {detection_output.total_inference_time_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)

    def switch_product(self, product_id: str) -> bool:
        success = self.algorithm_manager.set_current_product(product_id)
        if success:
            logger.info(f"Switched to product: {product_id}")
        else:
            logger.error(f"Failed to switch to product: {product_id}")
        return success

    def detect_image(self, image: np.ndarray, product_id: Optional[str] = None) -> Optional[DetectionOutput]:
        try:
            if product_id and product_id != self.algorithm_manager.current_product_id:
                self.switch_product(product_id)

            image_data = ImageData.create(
                camera_id="api",
                camera_position="api",
                image=image
            )

            return self.algorithm_manager.detect(image_data)

        except Exception as e:
            logger.error(f"Error in detect_image: {e}", exc_info=True)
            return None

    def _start_http_server(self):
        service = self
        api_cfg = self.config_manager.get_api_config()
        http_port = api_cfg.get("port", 8081)

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

                    elif path == "/api/products":
                        products = service.algorithm_manager.get_all_products()
                        self._send_json(200, {
                            "products": [
                                {"id": pid, "name": p.product_name}
                                for pid, p in products.items()
                            ],
                            "current_product": service.algorithm_manager.current_product_id
                        })

                    elif path == "/api/product/current":
                        product = service.algorithm_manager.get_product_config()
                        if product:
                            self._send_json(200, product.to_dict())
                        else:
                            self._send_json(404, {"error": "No product selected"})

                    elif path == "/api/algorithms":
                        algos = service.algorithm_manager.get_available_algorithms()
                        self._send_json(200, algos)

                    elif path == "/api/alerts":
                        limit = int(params.get("limit", [100])[0])
                        level = params.get("level", [None])[0]
                        alerts = service.alert_manager.get_recent_alerts(limit, level)
                        self._send_json(200, {"alerts": [a.__dict__ for a in alerts]})

                    elif path == "/api/stats":
                        self._send_json(200, service._get_stats())

                    elif path == "/api/alerts/reset-stop-line":
                        service.alert_manager.reset_stop_line()
                        self._send_json(200, {"status": "reset"})

                    elif path == "/api/alerts/clear-history":
                        service.alert_manager.clear_history()
                        self._send_json(200, {"status": "cleared"})

                    elif path == "/api/config":
                        self._send_json(200, service.config_manager.config)

                    elif path == "/api/reload-config":
                        service.config_manager.reload()
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

                    if path == "/api/product/switch":
                        product_id = data.get("product_id")
                        if product_id:
                            success = service.switch_product(product_id)
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing product_id"})

                    elif path == "/api/algorithm/params":
                        algo_type = data.get("algorithm_type")
                        params = data.get("params", {})
                        if algo_type:
                            success = service.algorithm_manager.reload_algorithm_params(
                                AlgorithmType(algo_type), params
                            )
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "Missing algorithm_type"})

                    elif path == "/api/detect":
                        image_base64 = data.get("image")
                        product_id = data.get("product_id")

                        if not image_base64:
                            self._send_json(400, {"error": "Missing image"})
                            return

                        import base64
                        img_bytes = base64.b64decode(image_base64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if image is None:
                            self._send_json(400, {"error": "Invalid image"})
                            return

                        result = service.detect_image(image, product_id)
                        if result:
                            response = result.to_dict()

                            if data.get("return_annotated", False):
                                product_config = service.algorithm_manager.get_product_config()
                                annotated = service.result_annotator.annotate(
                                    image, result.defects, product_config
                                )
                                _, img_encoded = cv2.imencode('.jpg', annotated)
                                response["annotated_image"] = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

                            self._send_json(200, response)
                        else:
                            self._send_json(500, {"error": "Detection failed"})

                    elif path == "/api/mq/reconnect":
                        success = service.message_consumer.reconnect() and service.result_producer.reconnect()
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
            "current_product": self.algorithm_manager.current_product_id,
            "consumer_connected": self.message_consumer.is_connected(),
            "producer_connected": self.result_producer.is_connected(),
            "stop_line_active": self.alert_manager.stop_line_active,
            "consecutive_ng_count": self.alert_manager.consecutive_ng_count,
            "products_count": len(self.algorithm_manager.get_all_products())
        }

    def _get_stats(self) -> Dict[str, Any]:
        return {
            "consumer": self.message_consumer.get_stats(),
            "producer": self.result_producer.get_stats(),
            "alerts": self.alert_manager.get_stats()
        }

    def wait(self):
        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.stop()


def main():
    config_path = os.environ.get("CONFIG_PATH", "./config/config.yaml")
    service = DefectDetectionService(config_path)

    try:
        service.start()
        service.wait()
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        service.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
