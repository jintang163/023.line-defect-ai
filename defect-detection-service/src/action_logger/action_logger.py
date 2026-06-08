from typing import Dict, Any, Optional, List, Callable
from queue import Queue
import threading
import time
from collections import deque
import json
import os

from src.utils.schemas import (
    ActionLogEntry, ActionLogType,
    DetectionOutput, PLCCommand, PLCCommandResult,
    AlertMessage, ManualOverrideRecord, ManualOverrideAction
)
from src.utils.logger import Logger

logger = Logger().logger

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logger.warning("sqlite3 not available, database logging disabled")

try:
    from influxdb import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logger.warning("influxdb not available, InfluxDB logging disabled")


class ActionLogger:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", True)
        self._max_memory_logs = config.get("max_memory_logs", 10000)
        self._enable_file_logging = config.get("enable_file_logging", True)
        self._enable_database_logging = config.get("enable_database_logging", False)
        self._enable_influxdb_logging = config.get("enable_influxdb_logging", False)

        self._log_file_path = config.get("log_file_path", "./data/action_logs.jsonl")
        self._sqlite_db_path = config.get("sqlite_db_path", "./data/action_logs.db")

        self._logs: deque = deque(maxlen=self._max_memory_logs)
        self._log_queue: Queue = Queue(maxsize=10000)
        self._callbacks: List[Callable[[ActionLogEntry], None]] = []
        self._lock = threading.RLock()

        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._influxdb_client: Optional[InfluxDBClient] = None

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._stats = {
            "total_logs": 0,
            "detection_logs": 0,
            "plc_command_logs": 0,
            "alert_logs": 0,
            "manual_override_logs": 0,
            "system_logs": 0,
            "file_writes": 0,
            "db_writes": 0,
            "influxdb_writes": 0,
            "failed_writes": 0
        }

        if self._enabled:
            self._init_storage()
            self._start_worker()

    def _init_storage(self):
        log_dir = os.path.dirname(self._log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        db_dir = os.path.dirname(self._sqlite_db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        if self._enable_database_logging and SQLITE_AVAILABLE:
            try:
                self._sqlite_conn = sqlite3.connect(self._sqlite_db_path, check_same_thread=False)
                self._init_sqlite_schema()
                logger.info(f"✅ SQLite action log database initialized: {self._sqlite_db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize SQLite: {e}", exc_info=True)
                self._enable_database_logging = False

        if self._enable_influxdb_logging and INFLUXDB_AVAILABLE:
            try:
                influx_config = self._config.get("influxdb", {})
                self._influxdb_client = InfluxDBClient(
                    host=influx_config.get("host", "localhost"),
                    port=influx_config.get("port", 8086),
                    username=influx_config.get("username", ""),
                    password=influx_config.get("password", ""),
                    database=influx_config.get("database", "defect_logs")
                )
                logger.info("✅ InfluxDB client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB: {e}", exc_info=True)
                self._enable_influxdb_logging = False

    def _init_sqlite_schema(self):
        if not self._sqlite_conn:
            return

        cursor = self._sqlite_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                log_id TEXT PRIMARY KEY,
                log_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                source TEXT NOT NULL,
                product_id TEXT,
                detection_id TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_type ON action_logs(log_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON action_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_id ON action_logs(product_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_id ON action_logs(detection_id)")

        self._sqlite_conn.commit()

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Action logger worker started")

    def _log_worker(self):
        while not self._stop_event.is_set():
            try:
                try:
                    log_entry = self._log_queue.get(timeout=0.1)
                except:
                    continue

                self._process_log_entry(log_entry)

            except Exception as e:
                logger.error(f"Action logger worker error: {e}", exc_info=True)
                time.sleep(1)

    def _process_log_entry(self, log_entry: ActionLogEntry):
        try:
            with self._lock:
                self._logs.append(log_entry)
                self._update_stats(log_entry)

            if self._enable_file_logging:
                self._write_to_file(log_entry)

            if self._enable_database_logging and self._sqlite_conn:
                self._write_to_sqlite(log_entry)

            if self._enable_influxdb_logging and self._influxdb_client:
                self._write_to_influxdb(log_entry)

            self._notify_callbacks(log_entry)

        except Exception as e:
            logger.error(f"Failed to process log entry: {e}", exc_info=True)
            self._stats["failed_writes"] += 1

    def _update_stats(self, log_entry: ActionLogEntry):
        self._stats["total_logs"] += 1
        type_map = {
            ActionLogType.DETECTION_RESULT: "detection_logs",
            ActionLogType.PLC_COMMAND: "plc_command_logs",
            ActionLogType.ALERT: "alert_logs",
            ActionLogType.MANUAL_OVERRIDE: "manual_override_logs",
            ActionLogType.SYSTEM: "system_logs"
        }
        stat_key = type_map.get(log_entry.log_type)
        if stat_key:
            self._stats[stat_key] += 1

    def _write_to_file(self, log_entry: ActionLogEntry):
        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                log_dict = log_entry.to_dict()
                f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
            self._stats["file_writes"] += 1
        except Exception as e:
            logger.warning(f"Failed to write log to file: {e}")
            self._stats["failed_writes"] += 1

    def _write_to_sqlite(self, log_entry: ActionLogEntry):
        try:
            cursor = self._sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO action_logs 
                (log_id, log_type, timestamp, level, message, source, product_id, detection_id, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.log_id,
                log_entry.log_type.value,
                log_entry.timestamp,
                log_entry.level,
                log_entry.message,
                log_entry.source,
                log_entry.product_id,
                log_entry.detection_id,
                json.dumps(log_entry.details, ensure_ascii=False)
            ))
            self._sqlite_conn.commit()
            self._stats["db_writes"] += 1
        except Exception as e:
            logger.warning(f"Failed to write log to SQLite: {e}")
            self._stats["failed_writes"] += 1

    def _write_to_influxdb(self, log_entry: ActionLogEntry):
        try:
            fields = {
                "message": log_entry.message,
                "detection_id": log_entry.detection_id,
            }

            for k, v in log_entry.details.items():
                if isinstance(v, (int, float)):
                    fields[k] = v
                elif isinstance(v, bool):
                    fields[k] = v
                elif isinstance(v, str):
                    fields[k] = v
                elif v is None:
                    continue
                else:
                    fields[k] = str(v)

            tags = {
                "log_type": log_entry.log_type.value,
                "level": log_entry.level,
                "source": log_entry.source,
            }

            if log_entry.product_id:
                tags["product_id"] = log_entry.product_id

            if log_entry.detection_id:
                tags["detection_id"] = log_entry.detection_id
                fields.pop("detection_id", None)

            json_body = [
                {
                    "measurement": "action_logs",
                    "tags": tags,
                    "time": int(log_entry.timestamp * 1e9),
                    "fields": fields
                }
            ]

            self._influxdb_client.write_points(json_body)
            self._stats["influxdb_writes"] += 1
        except Exception as e:
            logger.warning(f"Failed to write log to InfluxDB: {e}")
            self._stats["failed_writes"] += 1

    def _notify_callbacks(self, log_entry: ActionLogEntry):
        for callback in self._callbacks:
            try:
                callback(log_entry)
            except Exception as e:
                logger.error(f"Error in action log callback: {e}", exc_info=True)

    def log_detection_result(self, detection_output: DetectionOutput,
                             product_id: str = "") -> Optional[str]:
        if not self._enabled:
            return None

        result_icon = "OK" if detection_output.result.value == "OK" else "NG"
        level = "info" if detection_output.result.value == "OK" else "warning"

        defect_count = len(detection_output.defects)
        message = (f"检测结果: {result_icon} | "
                   f"缺陷数: {defect_count} | "
                   f"推理时间: {detection_output.total_inference_time_ms:.1f}ms")

        details = {
            "result": detection_output.result.value,
            "defect_count": defect_count,
            "inference_time_ms": detection_output.total_inference_time_ms,
            "alert_action": detection_output.alert_action.value,
            "defects": [d.to_dict() for d in detection_output.defects[:10]]
        }

        log_entry = ActionLogEntry.create(
            log_type=ActionLogType.DETECTION_RESULT,
            level=level,
            message=message,
            source="detection",
            product_id=product_id or detection_output.product_id,
            detection_id=detection_output.detection_id,
            details=details
        )

        self._log_queue.put(log_entry)
        return log_entry.log_id

    def log_plc_command(self, command: PLCCommand, result: Optional[PLCCommandResult] = None,
                        product_id: str = "", detection_id: str = "") -> Optional[str]:
        if not self._enabled:
            return None

        success = result.success if result else None
        level = "info" if success else "error" if success is not None else "info"
        status = "✅" if success else "❌" if success is False else "📤"

        message = (f"{status} PLC指令: {command.command_type.value} | "
                   f"线圈: {command.coil_address} | "
                   f"缺陷码: {command.defect_codes}")

        if result:
            message += f" | 响应时间: {result.response_time_ms:.1f}ms"
            if not result.success:
                message += f" | 错误: {result.error_message}"

        details = {
            "command": command.to_dict(),
            "result": result.to_dict() if result else None
        }

        log_entry = ActionLogEntry.create(
            log_type=ActionLogType.PLC_COMMAND,
            level=level,
            message=message,
            source="plc_connector",
            product_id=product_id,
            detection_id=detection_id or command.detection_id,
            details=details
        )

        self._log_queue.put(log_entry)
        return log_entry.log_id

    def log_alert(self, alert: AlertMessage, product_id: str = "") -> Optional[str]:
        if not self._enabled:
            return None

        level_map = {
            "info": "info",
            "warning": "warning",
            "error": "error",
            "critical": "critical"
        }
        level = level_map.get(alert.level, "info")

        message = (f"告警: {alert.category} | "
                   f"动作: {alert.action.value} | "
                   f"{alert.message}")

        details = alert.details

        log_entry = ActionLogEntry.create(
            log_type=ActionLogType.ALERT,
            level=level,
            message=message,
            source=alert.source,
            product_id=product_id,
            detection_id=alert.detection_id,
            details=details
        )

        self._log_queue.put(log_entry)
        return log_entry.log_id

    def log_manual_override(self, override: ManualOverrideRecord,
                            product_id: str = "") -> Optional[str]:
        if not self._enabled:
            return None

        action_label = {
            ManualOverrideAction.FORCE_PASS: "强制放行",
            ManualOverrideAction.FORCE_REJECT: "强制剔除",
            ManualOverrideAction.NORMAL: "正常"
        }.get(override.action, override.action.value)

        message = (f"人工干预: {action_label} | "
                   f"操作员: {override.operator} | "
                   f"原结果: {override.original_result.value} → "
                   f"新结果: {override.final_result.value} | "
                   f"原因: {override.reason}")

        details = {
            "operator": override.operator,
            "reason": override.reason,
            "original_result": override.original_result.value,
            "final_result": override.final_result.value,
            "details": override.details
        }

        log_entry = ActionLogEntry.create(
            log_type=ActionLogType.MANUAL_OVERRIDE,
            level="warning",
            message=message,
            source="manual_override",
            product_id=product_id,
            detection_id=override.detection_id,
            details=details
        )

        self._log_queue.put(log_entry)
        return log_entry.log_id

    def log_system_event(self, event: str, level: str = "info",
                         source: str = "system", details: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self._enabled:
            return None

        log_entry = ActionLogEntry.create(
            log_type=ActionLogType.SYSTEM,
            level=level,
            message=event,
            source=source,
            details=details or {}
        )

        self._log_queue.put(log_entry)
        return log_entry.log_id

    def register_callback(self, callback: Callable[[ActionLogEntry], None]):
        with self._lock:
            self._callbacks.append(callback)
            logger.info("Registered action log callback")

    def get_logs(self, log_type: Optional[ActionLogType] = None,
                 product_id: Optional[str] = None,
                 detection_id: Optional[str] = None,
                 level: Optional[str] = None,
                 limit: int = 100,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None) -> List[ActionLogEntry]:
        with self._lock:
            logs = list(self._logs)

        if log_type:
            logs = [l for l in logs if l.log_type == log_type]
        if product_id:
            logs = [l for l in logs if l.product_id == product_id]
        if detection_id:
            logs = [l for l in logs if l.detection_id == detection_id]
        if level:
            logs = [l for l in logs if l.level == level]
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]

        return logs[-limit:]

    def query_logs_from_db(self, log_type: Optional[str] = None,
                           product_id: Optional[str] = None,
                           detection_id: Optional[str] = None,
                           limit: int = 100,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self._enable_database_logging or not self._sqlite_conn:
            return []

        try:
            cursor = self._sqlite_conn.cursor()
            query = "SELECT * FROM action_logs WHERE 1=1"
            params = []

            if log_type:
                query += " AND log_type = ?"
                params.append(log_type)
            if product_id:
                query += " AND product_id = ?"
                params.append(product_id)
            if detection_id:
                query += " AND detection_id = ?"
                params.append(detection_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                log_dict = dict(zip(columns, row))
                if log_dict.get("details"):
                    try:
                        log_dict["details"] = json.loads(log_dict["details"])
                    except:
                        pass
                results.append(log_dict)

            return results

        except Exception as e:
            logger.error(f"Failed to query logs from DB: {e}", exc_info=True)
            return []

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "enabled": self._enabled,
                "memory_logs_count": len(self._logs),
                "queue_size": self._log_queue.qsize(),
                "file_logging": self._enable_file_logging,
                "database_logging": self._enable_database_logging,
                "influxdb_logging": self._enable_influxdb_logging,
                "log_file_path": self._log_file_path
            }

    def stop(self):
        logger.info("Stopping action logger...")
        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)

        while not self._log_queue.empty():
            try:
                log_entry = self._log_queue.get_nowait()
                self._process_log_entry(log_entry)
            except:
                break

        if self._sqlite_conn:
            try:
                self._sqlite_conn.close()
            except Exception as e:
                logger.warning(f"Error closing SQLite connection: {e}")

        if self._influxdb_client:
            try:
                self._influxdb_client.close()
            except Exception as e:
                logger.warning(f"Error closing InfluxDB client: {e}")

        logger.info("Action logger stopped")

    @property
    def enabled(self) -> bool:
        return self._enabled
