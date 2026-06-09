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

try:
    from src.data_management import DataManagementManager
    DATA_MANAGEMENT_AVAILABLE = True
except ImportError:
    DATA_MANAGEMENT_AVAILABLE = False

try:
    from src.system_monitor import SystemMonitorManager
    SYSTEM_MONITOR_AVAILABLE = True
except ImportError:
    SYSTEM_MONITOR_AVAILABLE = False

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
        alert_config = self.config_manager.get_alert_config()

        self.notification_dispatcher = None
        try:
            from src.notification.dispatcher import NotificationDispatcher
            notification_config = self.config_manager.get_notification_config()
            if notification_config.get("enable", False):
                self.notification_dispatcher = NotificationDispatcher(notification_config)
                logger.info("Notification dispatcher initialized")
        except ImportError:
            logger.warning("Notification module not available")

        self.alert_manager = AlertManager(
            max_history=1000,
            plc_connector=self.plc_connector,
            action_logger=self.action_logger,
            consecutive_ng_threshold=consecutive_threshold,
            auto_stop_line=auto_stop_line,
            notification_dispatcher=self.notification_dispatcher,
            alert_config=alert_config
        )

        if self.data_management_manager and alert_config.get("alert_history_db_enabled", True):
            try:
                record_store = self.data_management_manager.record_store
                if record_store and record_store._enabled:
                    from src.alert_manager import AlertGrade

                    def _on_alert_event(event_data):
                        try:
                            if isinstance(event_data, dict) and event_data.get("event") == "acknowledge":
                                record_store.update_alert_acknowledged(
                                    event_data["alert_id"],
                                    event_data.get("operator", "system")
                                )
                                return
                        except Exception:
                            pass

                        try:
                            if hasattr(event_data, 'to_dict'):
                                alert_dict = event_data.to_dict()
                                grade = self.alert_manager._grade_alert(event_data)
                                alert_dict["grade"] = grade.value
                                record_store.save_alert_event(alert_dict)
                        except Exception as e:
                            logger.warning(f"Failed to persist alert event: {e}")

                    self.alert_manager.register_event_callback(_on_alert_event)
                    logger.info("Alert event DB persistence enabled")
            except Exception as e:
                logger.warning(f"Failed to setup alert event persistence: {e}")

        self.production_tracker = None
        if PRODUCTION_TRACKER_AVAILABLE:
            production_config = self.config_manager.get_production_config()
            if production_config.get("enable", True):
                self.production_tracker = ProductionTracker(production_config)

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

        self.data_management_manager = None
        if DATA_MANAGEMENT_AVAILABLE:
            dm_config = self.config_manager.get_data_management_config()
            if dm_config.get("enable", False):
                self.data_management_manager = DataManagementManager(dm_config)
                logger.info("✅ 数据管理与追溯模块已启用")
            else:
                logger.info("数据管理与追溯模块已禁用")

        self.system_monitor_manager = None

        self.message_consumer = MessageConsumer(msg_config)
        self.result_producer = ResultProducer(msg_config)

        if SYSTEM_MONITOR_AVAILABLE:
            sm_config = self.config_manager.get_system_monitor_config()
            if sm_config.get("enable", False):
                self.system_monitor_manager = SystemMonitorManager(
                    config=sm_config,
                    algorithm_manager=self.algorithm_manager,
                    config_manager=self.config_manager,
                    message_consumer=self.message_consumer,
                    result_producer=self.result_producer,
                    data_management_manager=self.data_management_manager
                )
                logger.info("✅ 系统配置与监控模块已启用")
            else:
                logger.info("系统配置与监控模块已禁用")

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

        if self.system_monitor_manager and self.system_monitor_manager.enabled:
            self.system_monitor_manager.start()

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

        if self.data_management_manager:
            self.data_management_manager.close()
            logger.info("Data management manager stopped")

        if self.system_monitor_manager:
            self.system_monitor_manager.stop()
            logger.info("System monitor manager stopped")

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

            if self.alert_manager.stop_line_active:
                logger.warning(f"🛑 生产线已停机，跳过检测: {image_data.image_id}")
                if self.action_logger and self.action_logger.enabled:
                    self.action_logger.log_system_event(
                        event=f"生产线停机中，检测已跳过: {image_data.image_id}",
                        level="warning",
                        source="main",
                        details={"image_id": image_data.image_id, "sequence_id": image_data.sequence_id}
                    )
                return

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

            if self.data_management_manager and self.data_management_manager.enabled:
                self.data_management_manager.save_detection_result(
                    detection_output, product_config,
                    original_image=image_data.image,
                    annotated_image=annotated
                )

            if self.system_monitor_manager and self.system_monitor_manager.enabled:
                try:
                    _, frame_encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    camera_id = image_data.camera_id or "default"
                    self.system_monitor_manager.update_annotated_frame(camera_id, frame_encoded.tobytes())
                except Exception:
                    pass

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

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Max-Age", "86400")
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                try:
                    if path == "/" or path == "/index.html":
                        self._serve_static_file("frontend/index.html", "text/html")

                    elif path == "/monitor" or path == "/monitor.html":
                        self._serve_static_file("frontend/monitor.html", "text/html")

                    elif path.startswith("/frontend/"):
                        self._serve_frontend_file(path)

                    elif path == "/health":
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
                        self._send_json(200, {"alerts": [a.to_dict() for a in alerts]})

                    elif path == "/api/alerts/history":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            level = params.get("level", [None])[0]
                            category = params.get("category", [None])[0]
                            grade = params.get("grade", [None])[0]
                            ack_str = params.get("acknowledged", [None])[0]
                            acknowledged = None
                            if ack_str is not None:
                                acknowledged = ack_str.lower() == "true"
                            start_time = params.get("start_time", [None])[0]
                            end_time = params.get("end_time", [None])[0]
                            limit_val = int(params.get("limit", [100])[0])
                            offset_val = int(params.get("offset", [0])[0])
                            start_f = float(start_time) if start_time else None
                            end_f = float(end_time) if end_time else None

                            if service.data_management_manager and service.data_management_manager.record_store._enabled:
                                records = service.data_management_manager.record_store.query_alert_events(
                                    level=level, category=category, grade=grade,
                                    acknowledged=acknowledged, start_time=start_f, end_time=end_f,
                                    limit=limit_val, offset=offset_val
                                )
                                total = service.data_management_manager.record_store.count_alert_events(
                                    level=level, category=category, grade=grade,
                                    acknowledged=acknowledged, start_time=start_f, end_time=end_f
                                )
                                self._send_json(200, {"alerts": records, "total": total, "limit": limit_val, "offset": offset_val})
                            else:
                                alerts = service.alert_manager.get_alert_history(
                                    limit=limit_val, level=level, category=category,
                                    grade=grade, acknowledged=acknowledged,
                                    start_time=start_f, end_time=end_f
                                )
                                self._send_json(200, {"alerts": alerts, "total": len(alerts), "limit": limit_val, "offset": offset_val})

                    elif path == "/api/alerts/pending-urgent":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            pending = service.alert_manager.get_pending_urgent_alerts()
                            self._send_json(200, {"alerts": pending, "count": len(pending)})

                    elif path == "/api/alerts/relay-state":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            self._send_json(200, {
                                "relay_state": service.alert_manager.get_relay_state(),
                                "stop_line_active": service.alert_manager.stop_line_active
                            })

                    elif path == "/api/alerts/notification/config":
                        user = self._require_auth("full_config")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            if service.notification_dispatcher:
                                self._send_json(200, service.notification_dispatcher.get_config())
                            else:
                                self._send_json(200, {"enabled": False})

                    elif path == "/api/stats":
                        self._send_json(200, service._get_stats())

                    elif path == "/api/alerts/reset-stop-line":
                        service.alert_manager.reset_stop_line()
                        self._send_json(200, {"status": "reset"})

                    elif path == "/api/alerts/clear-history":
                        service.alert_manager.clear_history()
                        self._send_json(200, {"status": "cleared"})

                    elif path == "/api/config":
                        user = self._require_auth("full_config")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            self._send_json(200, service.config_manager.config)

                    elif path == "/api/reload-config":
                        user = self._require_auth("full_config")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
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

                    elif path == "/api/data-management/status":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            self._send_json(200, service.data_management_manager.get_stats())
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/search":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            product_id = params.get("product_id", [None])[0]
                            product_model = params.get("product_model", [None])[0]
                            start_time = params.get("start_time", [None])[0]
                            end_time = params.get("end_time", [None])[0]
                            defect_type = params.get("defect_type", [None])[0]
                            result = params.get("result", [None])[0]
                            limit = int(params.get("limit", [50])[0])
                            offset = int(params.get("offset", [0])[0])
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            search_result = service.data_management_manager.search(
                                product_id=product_id, product_model=product_model,
                                start_time=start_time_f, end_time=end_time_f,
                                defect_type=defect_type, result=result,
                                limit=limit, offset=offset
                            )
                            self._send_json(200, search_result)
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/record":
                        detection_id = params.get("detection_id", [None])[0]
                        if not detection_id:
                            self._send_json(400, {"error": "Missing detection_id"})
                        elif service.data_management_manager and service.data_management_manager.enabled:
                            record = service.data_management_manager.get_record_detail(detection_id)
                            if record:
                                self._send_json(200, record)
                            else:
                                self._send_json(404, {"error": "Record not found"})
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/suggestions":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            suggestions = service.data_management_manager.get_search_suggestions()
                            self._send_json(200, suggestions)
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/statistics/yield-trend":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            product_id = params.get("product_id", [None])[0]
                            start_time = params.get("start_time", [None])[0]
                            end_time = params.get("end_time", [None])[0]
                            interval = params.get("interval", ["hour"])[0]
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            result = service.data_management_manager.get_yield_trend(
                                product_id=product_id, start_time=start_time_f,
                                end_time=end_time_f, interval=interval
                            )
                            self._send_json(200, result)
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/statistics/defect-distribution":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            product_id = params.get("product_id", [None])[0]
                            start_time = params.get("start_time", [None])[0]
                            end_time = params.get("end_time", [None])[0]
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            result = service.data_management_manager.get_defect_distribution(
                                product_id=product_id, start_time=start_time_f,
                                end_time=end_time_f
                            )
                            self._send_json(200, result)
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/statistics/product-ranking":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            start_time = params.get("start_time", [None])[0]
                            end_time = params.get("end_time", [None])[0]
                            top_n = int(params.get("top_n", [10])[0])
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            result = service.data_management_manager.get_product_defect_ranking(
                                start_time=start_time_f, end_time=end_time_f, top_n=top_n
                            )
                            self._send_json(200, result)
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/statistics/overview":
                        if service.data_management_manager and service.data_management_manager.enabled:
                            start_time = params.get("start_time", [None])[0]
                            end_time = params.get("end_time", [None])[0]
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            result = service.data_management_manager.get_overview(
                                start_time=start_time_f, end_time=end_time_f
                            )
                            self._send_json(200, result)
                        else:
                            self._send_json(400, {"error": "Data management not available"})

                    elif path == "/api/data-management/download":
                        file_path_param = params.get("file", [None])[0]
                        if not file_path_param:
                            self._send_json(400, {"error": "Missing file parameter"})
                        elif not os.path.exists(file_path_param):
                            self._send_json(404, {"error": "File not found"})
                        else:
                            self._serve_download_file(file_path_param)

                    elif path == "/api/monitor/health":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            self._send_json(200, service.system_monitor_manager.get_health_status())
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/health/history":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            limit = int(params.get("limit", [100])[0])
                            history = service.system_monitor_manager.health_checker.get_history(limit)
                            self._send_json(200, {"history": history})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/frame":
                        camera_id = params.get("camera_id", ["default"])[0]
                        if service.system_monitor_manager and service.system_monitor_manager.enabled:
                            frame_data = service.system_monitor_manager.get_latest_frame(camera_id)
                            if frame_data:
                                self.send_response(200)
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Cache-Control", "no-cache")
                                self.end_headers()
                                self.wfile.write(frame_data)
                            else:
                                self._send_json(404, {"error": "No frame available"})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/cameras":
                        if service.system_monitor_manager and service.system_monitor_manager.enabled:
                            self._send_json(200, {"cameras": service.system_monitor_manager.get_all_camera_ids()})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/dashboard":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            self._send_json(200, service.system_monitor_manager.get_history_dashboard_data())
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/params":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            product_id = params.get("product_id", [None])[0]
                            if product_id and service.system_monitor_manager.param_adjuster:
                                params_data = service.system_monitor_manager.param_adjuster.get_product_params(product_id)
                                self._send_json(200, params_data or {})
                            else:
                                self._send_json(400, {"error": "Missing product_id"})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/params/log":
                        user = self._require_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            limit = int(params.get("limit", [100])[0])
                            log = service.system_monitor_manager.param_adjuster.get_change_log(limit)
                            self._send_json(200, {"change_log": log})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/users":
                        user = self._require_auth("full_config")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            self._send_json(200, {"users": service.system_monitor_manager.role_manager.list_users()})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

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

                    if path == "/api/monitor/auth/login":
                        if service.system_monitor_manager and service.system_monitor_manager.enabled:
                            username = data.get("username", "")
                            password = data.get("password", "")
                            token = service.system_monitor_manager.authenticate(username, password)
                            if token:
                                user_info = service.system_monitor_manager.verify_token(token)
                                self._send_json(200, {"token": token, "user": user_info})
                            else:
                                self._send_json(401, {"error": "Invalid credentials"})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/monitor/auth/verify":
                        token = data.get("token", "")
                        if service.system_monitor_manager and service.system_monitor_manager.enabled:
                            user_info = service.system_monitor_manager.verify_token(token)
                            if user_info:
                                self._send_json(200, {"valid": True, "user": user_info})
                            else:
                                self._send_json(200, {"valid": False})
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

                    elif path == "/api/product/switch":
                        user = self._require_write_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            product_id = data.get("product_id")
                            if product_id:
                                success = service.switch_product(product_id)
                                self._send_json(200, {"success": success})
                            else:
                                self._send_json(400, {"error": "Missing product_id"})

                    elif path == "/api/algorithm/params":
                        user = self._require_write_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
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
                        user = self._require_write_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            success = service.message_consumer.reconnect() and service.result_producer.reconnect()
                            self._send_json(200, {"success": success})

                    elif path == "/api/manual-override/apply":
                        user = self._require_write_auth("manual_override")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.manual_override_manager:
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
                        user = self._require_write_auth("manual_override")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            operator = data.get("operator", user.get("username", "system"))
                            reason = data.get("reason", "手动重置")
                            service.alert_manager.reset_stop_line(operator=operator, reason=reason)
                            self._send_json(200, {"status": "reset", "operator": operator, "reason": reason})

                    elif path == "/api/alerts/acknowledge":
                        user = self._require_write_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            alert_id = data.get("alert_id", "")
                            if not alert_id:
                                self._send_json(400, {"error": "Missing alert_id"})
                            else:
                                operator = data.get("operator", user.get("username", "system"))
                                success = service.alert_manager.acknowledge_alert(alert_id, operator=operator)
                                self._send_json(200, {"success": success, "alert_id": alert_id})

                    elif path == "/api/alerts/acknowledge-all":
                        user = self._require_write_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            operator = data.get("operator", user.get("username", "system"))
                            pending = service.alert_manager.get_pending_urgent_alerts()
                            count = 0
                            for a in pending:
                                aid = a.get("alert_id", "")
                                if aid and service.alert_manager.acknowledge_alert(aid, operator=operator):
                                    count += 1
                            self._send_json(200, {"success": True, "acknowledged_count": count})

                    elif path == "/api/alerts/export":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        else:
                            import csv
                            import io
                            import tempfile

                            level = data.get("level")
                            category = data.get("category")
                            grade = data.get("grade")
                            acknowledged = data.get("acknowledged")
                            start_time = data.get("start_time")
                            end_time = data.get("end_time")

                            if service.data_management_manager and service.data_management_manager.record_store._enabled:
                                alerts = service.data_management_manager.record_store.query_alert_events(
                                    level=level, category=category, grade=grade,
                                    acknowledged=acknowledged, start_time=start_time, end_time=end_time,
                                    limit=10000
                                )
                            else:
                                alerts = service.alert_manager.get_alert_history(
                                    limit=10000, level=level, category=category,
                                    grade=grade, acknowledged=acknowledged
                                )

                            output = io.StringIO()
                            if alerts:
                                fieldnames = list(alerts[0].keys())
                                writer = csv.DictWriter(output, fieldnames=fieldnames)
                                writer.writeheader()
                                for row in alerts:
                                    clean_row = {}
                                    for k, v in row.items():
                                        if isinstance(v, dict):
                                            clean_row[k] = json.dumps(v, ensure_ascii=False)
                                        elif isinstance(v, list):
                                            clean_row[k] = json.dumps(v, ensure_ascii=False)
                                        else:
                                            clean_row[k] = v
                                    writer.writerow(clean_row)

                            tmp = tempfile.NamedTemporaryFile(
                                mode='w', suffix='.csv', delete=False,
                                dir=service.config_manager.get_data_management_config().get("export", {}).get("temp_dir", "./data/export_temp"),
                                encoding='utf-8-sig'
                            )
                            tmp.write(output.getvalue())
                            tmp.close()
                            self._send_json(200, {
                                "file": os.path.basename(tmp.name),
                                "count": len(alerts)
                            })

                    elif path == "/api/alerts/notification/receivers":
                        user = self._require_write_auth("full_config")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.notification_dispatcher or not service.notification_dispatcher.enabled:
                            self._send_json(400, {"error": "Notification not available"})
                        else:
                            group_name = data.get("group", "")
                            channel_name = data.get("channel", "")
                            receivers = data.get("receivers", [])
                            if not group_name or not channel_name:
                                self._send_json(400, {"error": "Missing group or channel"})
                            else:
                                service.notification_dispatcher.update_receivers(group_name, channel_name, receivers)
                                self._send_json(200, {"success": True})

                    elif path == "/api/plc/command/reject":
                        user = self._require_write_auth("manual_override")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                        else:
                            detection_id = data.get("detection_id", "")
                            command_id = service.plc_connector.send_reject_command(
                                detection_id=detection_id,
                                defect_types=None,
                                alert_action=AlertAction.REJECT
                            )
                            self._send_json(200, {"command_id": command_id})

                    elif path == "/api/plc/command/stop-line":
                        user = self._require_write_auth("manual_override")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                        else:
                            detection_id = data.get("detection_id", "")
                            reason = data.get("reason", "手动触发")
                            command_id = service.plc_connector.send_stop_line_command(
                                detection_id=detection_id,
                                reason=reason
                            )
                            self._send_json(200, {"command_id": command_id})

                    elif path == "/api/plc/command/reset":
                        user = self._require_write_auth("manual_override")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                        else:
                            command_id = service.plc_connector.send_reset_command()
                            self._send_json(200, {"command_id": command_id})

                    elif path == "/api/plc/command/alarm":
                        user = self._require_write_auth("manual_override")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.plc_connector or not service.plc_connector.enabled:
                            self._send_json(400, {"error": "PLC connector not available or disabled"})
                        else:
                            detection_id = data.get("detection_id", "")
                            alarm_type = data.get("alarm_type", "manual")
                            command_id = service.plc_connector.send_alarm_command(
                                detection_id=detection_id,
                                alarm_type=alarm_type
                            )
                            self._send_json(200, {"command_id": command_id})

                    elif path == "/api/data-management/export/images":
                        user = self._require_write_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.data_management_manager or not service.data_management_manager.enabled:
                            self._send_json(400, {"error": "Data management not available"})
                        else:
                            detection_ids = data.get("detection_ids", [])
                            output_filename = data.get("output_filename")
                            if not detection_ids:
                                self._send_json(400, {"error": "Missing detection_ids"})
                            else:
                                zip_path = service.data_management_manager.export_images_zip(
                                    detection_ids, output_filename
                                )
                                if zip_path:
                                    self._send_json(200, {"file_path": zip_path})
                                else:
                                    self._send_json(500, {"error": "Export failed"})

                    elif path == "/api/data-management/export/records":
                        user = self._require_write_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.data_management_manager or not service.data_management_manager.enabled:
                            self._send_json(400, {"error": "Data management not available"})
                        else:
                            detection_ids = data.get("detection_ids")
                            product_id = data.get("product_id")
                            start_time = data.get("start_time")
                            end_time = data.get("end_time")
                            result_filter = data.get("result")
                            defect_type = data.get("defect_type")
                            output_filename = data.get("output_filename")
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            csv_path = service.data_management_manager.export_records_excel(
                                detection_ids=detection_ids, product_id=product_id,
                                start_time=start_time_f, end_time=end_time_f,
                                result=result_filter, defect_type=defect_type,
                                output_filename=output_filename
                            )
                            if csv_path:
                                self._send_json(200, {"file_path": csv_path})
                            else:
                                self._send_json(500, {"error": "Export failed"})

                    elif path == "/api/data-management/export/all":
                        user = self._require_write_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.data_management_manager or not service.data_management_manager.enabled:
                            self._send_json(400, {"error": "Data management not available"})
                        else:
                            detection_ids = data.get("detection_ids")
                            product_id = data.get("product_id")
                            start_time = data.get("start_time")
                            end_time = data.get("end_time")
                            result_filter = data.get("result")
                            defect_type = data.get("defect_type")
                            output_filename = data.get("output_filename")
                            start_time_f = float(start_time) if start_time else None
                            end_time_f = float(end_time) if end_time else None
                            zip_path = service.data_management_manager.export_all(
                                detection_ids=detection_ids, product_id=product_id,
                                start_time=start_time_f, end_time=end_time_f,
                                result=result_filter, defect_type=defect_type,
                                output_filename=output_filename
                            )
                            if zip_path:
                                self._send_json(200, {"file_path": zip_path})
                            else:
                                self._send_json(500, {"error": "Export failed"})

                    elif path == "/api/monitor/auth/users":
                        user = self._require_write_auth("full_config")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.system_monitor_manager or not service.system_monitor_manager.enabled:
                            self._send_json(400, {"error": "System monitor not available"})
                        else:
                            action = data.get("action", "list")
                            if action == "add":
                                username = data.get("username", "")
                                password = data.get("password", "")
                                role = data.get("role", "operator")
                                if username and password:
                                    success = service.system_monitor_manager.role_manager.add_user(username, password, role)
                                    self._send_json(200, {"success": success})
                                else:
                                    self._send_json(400, {"error": "Missing username or password"})
                            elif action == "remove":
                                username = data.get("username", "")
                                if username:
                                    success = service.system_monitor_manager.role_manager.remove_user(username)
                                    self._send_json(200, {"success": success})
                                else:
                                    self._send_json(400, {"error": "Missing username"})
                            else:
                                users = service.system_monitor_manager.role_manager.list_users()
                                self._send_json(200, {"users": users})

                    elif path == "/api/monitor/params/adjust":
                        user = self._require_write_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.system_monitor_manager or not service.system_monitor_manager.enabled:
                            self._send_json(400, {"error": "System monitor not available"})
                        else:
                            product_id = data.get("product_id", "")
                            param_path = data.get("param_path", "")
                            new_value = data.get("new_value")
                            operator = data.get("operator", user.get("username", "system"))
                            if not product_id or not param_path or new_value is None:
                                self._send_json(400, {"error": "Missing product_id, param_path, or new_value"})
                            else:
                                success = service.system_monitor_manager.param_adjuster.adjust_product_param(
                                    product_id, param_path, new_value, operator
                                )
                                self._send_json(200, {"success": success})

                    elif path == "/api/monitor/params/threshold":
                        user = self._require_write_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.system_monitor_manager or not service.system_monitor_manager.enabled:
                            self._send_json(400, {"error": "System monitor not available"})
                        else:
                            product_id = data.get("product_id", "")
                            defect_type = data.get("defect_type", "")
                            field = data.get("field", "")
                            new_value = data.get("new_value")
                            operator = data.get("operator", user.get("username", "system"))
                            if not all([product_id, defect_type, field, new_value is not None]):
                                self._send_json(400, {"error": "Missing required fields"})
                            else:
                                success = service.system_monitor_manager.param_adjuster.adjust_threshold(
                                    product_id, defect_type, field, new_value, operator
                                )
                                self._send_json(200, {"success": success})

                    elif path == "/api/monitor/params/rollback":
                        user = self._require_write_auth("adjust_params")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif not service.system_monitor_manager or not service.system_monitor_manager.enabled:
                            self._send_json(400, {"error": "System monitor not available"})
                        else:
                            change_id = data.get("change_id")
                            if change_id is None:
                                self._send_json(400, {"error": "Missing change_id"})
                            else:
                                success = service.system_monitor_manager.param_adjuster.rollback_change(change_id)
                                self._send_json(200, {"success": success})

                    elif path == "/api/monitor/health/check":
                        user = self._require_auth("view")
                        if not user:
                            self._send_json(401, {"error": "Authentication required"})
                        elif service.system_monitor_manager and service.system_monitor_manager.enabled:
                            result = service.system_monitor_manager.get_health_status()
                            self._send_json(200, result)
                        else:
                            self._send_json(400, {"error": "System monitor not available"})

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

            def _extract_bearer_token(self) -> Optional[str]:
                auth_header = self.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    return auth_header[7:].strip()
                return None

            def _require_auth(self, permission: str = "view") -> Optional[Dict]:
                if not service.system_monitor_manager or not service.system_monitor_manager.enabled:
                    return {"username": "anonymous", "role": "admin", "permissions": ["view", "manual_override", "adjust_params", "full_config"]}
                token = self._extract_bearer_token()
                if not token:
                    return None
                user_info = service.system_monitor_manager.verify_token(token)
                if not user_info:
                    return None
                if permission not in user_info.get("permissions", []):
                    return None
                return user_info

            def _require_write_auth(self, permission: str = "adjust_params") -> Optional[Dict]:
                if not service.system_monitor_manager or not service.system_monitor_manager.enabled:
                    return {"username": "anonymous", "role": "admin", "permissions": ["view", "manual_override", "adjust_params", "full_config"]}
                token = self._extract_bearer_token()
                if not token:
                    return None
                user_info = service.system_monitor_manager.verify_token(token)
                if not user_info:
                    return None
                if permission not in user_info.get("permissions", []):
                    return None
                return user_info

            def _serve_static_file(self, relative_path: str, content_type: str):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                file_path = os.path.join(base_dir, relative_path)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self._send_json(404, {"error": "File not found"})

            def _serve_frontend_file(self, request_path: str):
                mime_types = {
                    ".html": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".json": "application/json",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".svg": "image/svg+xml",
                    ".ico": "image/x-icon"
                }
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                relative = request_path.lstrip("/")
                file_path = os.path.join(base_dir, relative)
                file_path = os.path.normpath(file_path)
                if not file_path.startswith(base_dir):
                    self._send_json(403, {"error": "Forbidden"})
                    return
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    content_type = mime_types.get(ext, "application/octet-stream")
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self._send_json(404, {"error": "File not found"})

            def _serve_download_file(self, file_path: str):
                abs_path = os.path.abspath(file_path)
                if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
                    self._send_json(404, {"error": "File not found"})
                    return
                filename = os.path.basename(abs_path)
                ext = os.path.splitext(filename)[1].lower()
                mime_types = {".zip": "application/zip", ".csv": "text/csv", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
                content_type = mime_types.get(ext, "application/octet-stream")
                file_size = os.path.getsize(abs_path)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(file_size))
                self.end_headers()
                with open(abs_path, "rb") as f:
                    self.wfile.write(f.read())

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
            },
            "data_management": {
                "enabled": self.data_management_manager is not None and self.data_management_manager.enabled
            },
            "system_monitor": {
                "enabled": self.system_monitor_manager is not None and self.system_monitor_manager.enabled
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

        if self.data_management_manager and self.data_management_manager.enabled:
            stats["data_management"] = self.data_management_manager.get_stats()

        if self.system_monitor_manager and self.system_monitor_manager.enabled:
            stats["system_monitor"] = self.system_monitor_manager.get_stats()

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
