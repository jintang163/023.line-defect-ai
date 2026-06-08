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

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import ConfigManager
from src.utils.logger import Logger
from src.utils.schemas import (
    ImageData, DetectionOutput, ProductConfig, AlgorithmType,
    ManualOverrideAction, DetectionResult, AlertAction,
    YieldSnapshot
)
from src.algorithm_manager import AlgorithmManager
from src.result_annotator import ResultAnnotator
from src.alert_manager import AlertManager
from src.messaging.message_consumer import MessageConsumer
from src.messaging.result_producer import ResultProducer

try:
    from src.plc.plc_connector import PLCConnector
    PLC_AVAILABLE = True
except ImportError:
    PLC_AVAILABLE = False

try:
    from src.action_logger.action_logger import ActionLogger
    ACTION_LOGGER_AVAILABLE = True
except ImportError:
    ACTION_LOGGER_AVAILABLE = False

try:
    from src.production.production_tracker import ProductionTracker
    PRODUCTION_TRACKER_AVAILABLE = True
except ImportError:
    PRODUCTION_TRACKER_AVAILABLE = False

try:
    from src.manual_override.manual_override_manager import ManualOverrideManager
    MANUAL_OVERRIDE_AVAILABLE = True
except ImportError:
    MANUAL_OVERRIDE_AVAILABLE = False

logger = Logger("defect-detection-service", "INFO", "./logs/defect-detection.log").logger


class DefectDetectionService:
    def __init__(self, config_path: str = "./config/config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self._is_running = False
        self._shutdown_event = threading.Event()

        self._yield_db_conn: Optional[sqlite3.Connection] = None
        self._yield_db_enabled = False
        self._yield_api_enabled = False

        self._init_components()
        self._init_callbacks()

    def _init_components(self):
        logger.info("Initializing defect detection service components...")

        msg_config = self.config_manager.get_messaging_config()
        enable_parallel = msg_config.get("enable_parallel_processing", True)
        max_workers = msg_config.get("max_parallel_workers", 4)

        self.algorithm_manager = AlgorithmManager(
            enable_parallel=enable_parallel,
            max_workers=max_workers
        )

        products_config_path = self.config_manager.get_products_config_path()
        if not self.algorithm_manager.load_products_config(products_config_path):
            logger.warning("Failed to load products configuration")

        self.result_annotator = ResultAnnotator()

        self.action_logger = None
        if ACTION_LOGGER_AVAILABLE:
            action_log_config = self.config_manager.get_action_log_config()
            if action_log_config.get("enable", True):
                self.action_logger = ActionLogger(action_log_config)

        self.plc_connector = None
        if PLC_AVAILABLE:
            plc_config = self.config_manager.get_plc_config()
            if plc_config.get("enable", False):
                self.plc_connector = PLCConnector(plc_config)

                if self.action_logger and self.action_logger.enabled:
                    def on_plc_command_result(command, result):
                        self.action_logger.log_plc_command(command, result)
                    self.plc_connector.register_result_callback(on_plc_command_result)

        consecutive_threshold = self.config_manager.get_consecutive_ng_threshold()
        auto_stop_line = self.config_manager.get_auto_stop_line()

        self.alert_manager = AlertManager(
            max_history=1000,
            plc_connector=self.plc_connector,
            action_logger=self.action_logger,
            consecutive_ng_threshold=consecutive_threshold,
            auto_stop_line=auto_stop_line
        )

        self.production_tracker = None
        if PRODUCTION_TRACKER_AVAILABLE:
            production_config = self.config_manager.get_production_config()
            if production_config.get("enable", True):
                self.production_tracker = ProductionTracker(production_config)

                if self.plc_connector and self.plc_connector.enabled:
                    def on_emergency_stop(count, reason):
                        self.plc_connector.send_stop_line_command("", reason)
                    self.production_tracker.register_emergency_stop_callback(on_emergency_stop)

                self._init_yield_persistence(production_config)

                if self._yield_db_enabled or self._yield_api_enabled:
                    def on_snapshot_upload(snapshot):
                        self._persist_yield_snapshot(snapshot)
                    self.production_tracker.register_snapshot_callback(on_snapshot_upload)

        self.manual_override_manager = None
        if MANUAL_OVERRIDE_AVAILABLE:
            override_config = self.config_manager.get_manual_override_config()
            max_history = override_config.get("max_history", 10000)
            self.manual_override_manager = ManualOverrideManager(
                max_history=max_history,
                plc_connector=self.plc_connector
            )

            if self.action_logger and self.action_logger.enabled:
                def on_manual_override(record):
                    self.action_logger.log_manual_override(record)
                self.manual_override_manager.register_override_callback(on_manual_override)

        self.message_consumer = MessageConsumer(msg_config)
        self.result_producer = ResultProducer(msg_config)

        logger.info("All components initialized")

    def _init_yield_persistence(self, production_config: Dict[str, Any]):
        self._yield_db_enabled = production_config.get("enable_database_persistence", True) and SQLITE_AVAILABLE
        self._yield_api_enabled = production_config.get("enable_api_upload", False) and REQUESTS_AVAILABLE

        if self._yield_db_enabled:
            db_path = production_config.get("sqlite_db_path", "./data/yield_snapshots.db")
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            try:
                self._yield_db_conn = sqlite3.connect(db_path, check_same_thread=False)
                self._init_yield_db_schema()
                logger.info(f"✅ 良率快照数据库已初始化: {db_path}")
            except Exception as e:
                logger.error(f"初始化良率快照数据库失败: {e}", exc_info=True)
                self._yield_db_enabled = False

        if self._yield_api_enabled:
            api_url = production_config.get("api_upload_url", "")
            if not api_url:
                logger.warning("API上传URL未配置，已禁用API上传")
                self._yield_api_enabled = False
            else:
                logger.info(f"✅ 良率快照API上传已启用: {api_url}")

    def _init_yield_db_schema(self):
        if not self._yield_db_conn:
            return

        cursor = self._yield_db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS yield_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                total_count INTEGER NOT NULL,
                ok_count INTEGER NOT NULL,
                ng_count INTEGER NOT NULL,
                yield_rate REAL NOT NULL,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL,
                defect_distribution TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_product_id ON yield_snapshots(product_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp ON yield_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_yield_rate ON yield_snapshots(yield_rate)")

        self._yield_db_conn.commit()

    def _persist_yield_snapshot(self, snapshot: YieldSnapshot):
        try:
            if self._yield_db_enabled and self._yield_db_conn:
                self._save_snapshot_to_db(snapshot)

            if self._yield_api_enabled:
                self._upload_snapshot_to_api(snapshot)

        except Exception as e:
            logger.error(f"持久化良率快照失败: {e}", exc_info=True)

    def _save_snapshot_to_db(self, snapshot: YieldSnapshot):
        if not self._yield_db_conn:
            return

        try:
            cursor = self._yield_db_conn.cursor()
            cursor.execute("""
                INSERT INTO yield_snapshots 
                (snapshot_id, product_id, timestamp, total_count, ok_count, ng_count, 
                 yield_rate, period_start, period_end, defect_distribution, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id,
                snapshot.product_id,
                snapshot.timestamp,
                snapshot.total_count,
                snapshot.ok_count,
                snapshot.ng_count,
                snapshot.yield_rate,
                snapshot.period_start,
                snapshot.period_end,
                json.dumps(snapshot.defect_distribution, ensure_ascii=False),
                json.dumps(snapshot.details, ensure_ascii=False)
            ))
            self._yield_db_conn.commit()
            logger.info(f"💾 良率快照已保存到数据库: {snapshot.snapshot_id}")
        except Exception as e:
            logger.warning(f"保存良率快照到数据库失败: {e}")

    def _upload_snapshot_to_api(self, snapshot: YieldSnapshot):
        if not REQUESTS_AVAILABLE:
            return

        production_config = self.config_manager.get_production_config()
        api_url = production_config.get("api_upload_url", "")
        timeout = production_config.get("api_upload_timeout_ms", 5000) / 1000.0

        if not api_url:
            return

        try:
            payload = snapshot.to_dict()
            response = requests.post(api_url, json=payload, timeout=timeout)
            if response.status_code == 200:
                logger.info(f"🌐 良率快照已上传到API: {snapshot.snapshot_id}")
            else:
                logger.warning(f"上传良率快照到API失败: HTTP {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"上传良率快照到API失败: {e}")

    def _init_callbacks(self):
        self.message_consumer.set_callback(self._on_image_received)
        self.message_consumer.set_product_switch_callback(self.switch_product)

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

        if self.plc_connector:
            self.plc_connector.disconnect()
            logger.info("PLC connector stopped")

        if self.action_logger:
            self.action_logger.stop()
            logger.info("Action logger stopped")

        if self._yield_db_conn:
            try:
                self._yield_db_conn.close()
                logger.info("Yield snapshot database connection closed")
            except Exception as e:
                logger.warning(f"Error closing yield snapshot database: {e}")

        if hasattr(self, '_http_server'):
            self._http_server.shutdown()

        self._is_running = False
        logger.info("Defect detection service stopped")

    def _on_image_received(self, image_data: ImageData):
        try:
            logger.debug(f"Processing image: {image_data.image_id} from {image_data.camera_id}")

            detection_output = self.algorithm_manager.detect(image_data)

            product_config = self.algorithm_manager.get_product_config()

            if self.manual_override_manager and self.manual_override_manager.has_override(image_data.sequence_id):
                override = self.manual_override_manager.get_override(image_data.sequence_id)
                if override:
                    original_result = detection_output.result
                    detection_output.result = override.final_result
                    detection_output.metadata["manual_override"] = override.to_dict()
                    logger.info(f"🔧 人工干预: {original_result.value} → {override.final_result.value} | "
                               f"操作员: {override.operator} | 原因: {override.reason}")

            if product_config:
                alerts = self.alert_manager.process_detection_result(
                    detection_output, product_config
                )
                for alert in alerts:
                    logger.info(f"Alert generated: {alert.level} - {alert.message}")

            if self.production_tracker and product_config:
                is_emergency_stop, snapshot = self.production_tracker.process_result(
                    detection_output, product_config
                )
                if snapshot:
                    logger.info(f"📊 良率快照生成: {snapshot.yield_rate:.2f}% | "
                               f"OK: {snapshot.ok_count}/{snapshot.total_count}")

            annotated = self.result_annotator.annotate(
                image_data.image,
                detection_output.defects,
                product_config
            )
            detection_output.annotated_image = annotated

            if self.action_logger and self.action_logger.enabled:
                self.action_logger.log_detection_result(detection_output)

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

            if self.production_tracker:
                self.production_tracker.reset_stats(product_id)
                logger.info(f"生产统计已重置为产品: {product_id}")

            if self.action_logger and self.action_logger.enabled:
                self.action_logger.log_system_event(
                    event=f"产品切换: {product_id}",
                    level="info",
                    source="main",
                    details={"product_id": product_id}
                )
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

                    elif path == "/api/plc/status":
                        plc_status = {
                            "enabled": service.plc_connector.enabled if service.plc_connector else False,
                            "connected": service.plc_connector.is_connected() if service.plc_connector else False,
                            "stats": service.plc_connector.get_stats() if service.plc_connector else {}
                        }
                        self._send_json(200, plc_status)

                    elif path == "/api/plc/connect":
                        if service.plc_connector:
                            success = service.plc_connector.connect()
                            self._send_json(200, {"success": success})
                        else:
                            self._send_json(400, {"error": "PLC connector not available"})

                    elif path == "/api/plc/disconnect":
                        if service.plc_connector:
                            service.plc_connector.disconnect()
                            self._send_json(200, {"status": "disconnected"})
                        else:
                            self._send_json(400, {"error": "PLC connector not available"})

                    elif path == "/api/production/stats":
                        if service.production_tracker:
                            product_id = params.get("product_id", [None])[0]
                            stats = service.production_tracker.get_stats(product_id)
                            self._send_json(200, stats)
                        else:
                            self._send_json(400, {"error": "Production tracker not available"})

                    elif path == "/api/production/snapshots":
                        if service.production_tracker:
                            product_id = params.get("product_id", [None])[0]
                            limit = int(params.get("limit", [100])[0])
                            snapshots = service.production_tracker.get_snapshots(product_id, limit)
                            self._send_json(200, {"snapshots": [s.to_dict() for s in snapshots]})
                        else:
                            self._send_json(400, {"error": "Production tracker not available"})

                    elif path == "/api/production/defect-distribution":
                        if service.production_tracker:
                            product_id = params.get("product_id", [None])[0]
                            dist = service.production_tracker.get_defect_distribution(product_id)
                            self._send_json(200, {"distribution": dist})
                        else:
                            self._send_json(400, {"error": "Production tracker not available"})

                    elif path == "/api/production/reset":
                        if service.production_tracker:
                            product_id = params.get("product_id", [None])[0]
                            service.production_tracker.reset_stats(product_id)
                            self._send_json(200, {"status": "reset"})
                        else:
                            self._send_json(400, {"error": "Production tracker not available"})

                    elif path == "/api/manual-override/history":
                        if service.manual_override_manager:
                            operator = params.get("operator", [None])[0]
                            action = params.get("action", [None])[0]
                            limit = int(params.get("limit", [100])[0])
                            action_enum = ManualOverrideAction(action) if action else None
                            records = service.manual_override_manager.get_overrides(operator, action_enum, limit)
                            self._send_json(200, {"records": [r.to_dict() for r in records]})
                        else:
                            self._send_json(400, {"error": "Manual override manager not available"})

                    elif path == "/api/manual-override/stats":
                        if service.manual_override_manager:
                            stats = service.manual_override_manager.get_stats()
                            self._send_json(200, stats)
                        else:
                            self._send_json(400, {"error": "Manual override manager not available"})

                    elif path == "/api/action-logs":
                        if service.action_logger:
                            log_type = params.get("log_type", [None])[0]
                            product_id = params.get("product_id", [None])[0]
                            detection_id = params.get("detection_id", [None])[0]
                            level = params.get("level", [None])[0]
                            limit = int(params.get("limit", [100])[0])
                            from src.utils.schemas import ActionLogType
                            log_type_enum = ActionLogType(log_type) if log_type else None
                            logs = service.action_logger.get_logs(log_type_enum, product_id, detection_id, level, limit)
                            self._send_json(200, {"logs": [l.to_dict() for l in logs]})
                        else:
                            self._send_json(400, {"error": "Action logger not available"})

                    elif path == "/api/action-logs/stats":
                        if service.action_logger:
                            stats = service.action_logger.get_stats()
                            self._send_json(200, stats)
                        else:
                            self._send_json(400, {"error": "Action logger not available"})

                    elif path == "/api/action-logs/query":
                        if service.action_logger:
                            log_type = params.get("log_type", [None])[0]
                            product_id = params.get("product_id", [None])[0]
                            detection_id = params.get("detection_id", [None])[0]
                            limit = int(params.get("limit", [100])[0])
                            logs = service.action_logger.query_logs_from_db(
                                log_type, product_id, detection_id, limit
                            )
                            self._send_json(200, {"logs": logs})
                        else:
                            self._send_json(400, {"error": "Action logger not available"})

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

                    elif path == "/api/manual-override/apply":
                        if not service.manual_override_manager:
                            self._send_json(400, {"error": "Manual override manager not available"})
                            return

                        sequence_id = data.get("sequence_id")
                        detection_id = data.get("detection_id")
                        action = data.get("action")
                        operator = data.get("operator", "unknown")
                        reason = data.get("reason", "")
                        original_result_str = data.get("original_result")
                        final_result_str = data.get("final_result")
                        details = data.get("details", {})

                        if not sequence_id or not detection_id or not action or not original_result_str or not final_result_str:
                            self._send_json(400, {"error": "Missing required fields: sequence_id, detection_id, action, original_result, final_result"})
                            return

                        try:
                            action_enum = ManualOverrideAction(action)
                            original_result = DetectionResult(original_result_str)
                            final_result = DetectionResult(final_result_str)
                        except Exception as e:
                            self._send_json(400, {"error": f"Invalid enum value: {e}"})
                            return

                        record = service.manual_override_manager.apply_override(
                            sequence_id=sequence_id,
                            detection_id=detection_id,
                            action=action_enum,
                            operator=operator,
                            reason=reason,
                            original_result=original_result,
                            final_result=final_result,
                            details=details
                        )

                        if record:
                            self._send_json(200, {"success": True, "record": record.to_dict()})
                        else:
                            self._send_json(500, {"error": "Failed to apply override"})

                    elif path == "/api/alerts/reset-stop-line":
                        operator = data.get("operator", "system")
                        reason = data.get("reason", "手动重置")
                        service.alert_manager.reset_stop_line(operator=operator, reason=reason)
                        self._send_json(200, {"status": "reset", "operator": operator, "reason": reason})

                    elif path == "/api/plc/command/reject":
                        if not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                            return

                        detection_id = data.get("detection_id", "")
                        command_id = service.plc_connector.send_reject_command(
                            detection_id=detection_id,
                            defect_types=None,
                            alert_action=AlertAction.REJECT
                        )
                        self._send_json(200, {"command_id": command_id})

                    elif path == "/api/plc/command/stop-line":
                        if not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                            return

                        detection_id = data.get("detection_id", "")
                        reason = data.get("reason", "手动触发")
                        command_id = service.plc_connector.send_stop_line_command(
                            detection_id=detection_id,
                            reason=reason
                        )
                        self._send_json(200, {"command_id": command_id})

                    elif path == "/api/plc/command/reset":
                        if not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                            return

                        command_id = service.plc_connector.send_reset_command()
                        self._send_json(200, {"command_id": command_id})

                    elif path == "/api/plc/command/alarm":
                        if not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                            return

                        detection_id = data.get("detection_id", "")
                        alarm_type = data.get("alarm_type", "manual")
                        command_id = service.plc_connector.send_alarm_command(
                            detection_id=detection_id,
                            alarm_type=alarm_type
                        )
                        self._send_json(200, {"command_id": command_id})

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
            "products_count": len(self.algorithm_manager.get_all_products()),
            "plc": {
                "enabled": self.plc_connector.enabled if self.plc_connector else False,
                "connected": self.plc_connector.is_connected() if self.plc_connector else False
            },
            "action_logger": {
                "enabled": self.action_logger.enabled if self.action_logger else False
            },
            "production_tracker": {
                "enabled": self.production_tracker is not None
            },
            "manual_override": {
                "enabled": self.manual_override_manager is not None
            }
        }

    def _get_stats(self) -> Dict[str, Any]:
        stats = {
            "consumer": self.message_consumer.get_stats(),
            "producer": self.result_producer.get_stats(),
            "alerts": self.alert_manager.get_stats()
        }

        if self.plc_connector:
            stats["plc"] = self.plc_connector.get_stats()

        if self.action_logger:
            stats["action_logger"] = self.action_logger.get_stats()

        if self.production_tracker:
            product_id = self.algorithm_manager.current_product_id
            stats["production"] = self.production_tracker.get_stats(product_id)

        if self.manual_override_manager:
            stats["manual_override"] = self.manual_override_manager.get_stats()

        return stats

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
