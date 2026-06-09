from typing import Dict, Any, Optional, List, Tuple
import threading
import os
import json
from datetime import datetime

from src.utils.schemas import DetectionRecord, DetectionResult
from src.utils.logger import Logger

logger = Logger("record_store", "INFO", "./logs/defect-detection.log").logger

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logger.warning("sqlite3 not available, record store disabled")


class RecordStore:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)
        self._sqlite_db_dir = config.get("sqlite_db_dir", "./data")
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

        if self._enabled and SQLITE_AVAILABLE:
            self._init_db()

    def _init_db(self):
        db_dir = self._sqlite_db_dir
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        db_path = os.path.join(db_dir, "detection_records.db")
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            logger.info(f"SQLite record store initialized: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite record store: {e}", exc_info=True)
            self._enabled = False

    def _ensure_table(self, table_name: str):
        if not self._conn:
            return
        cursor = self._conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                record_id TEXT PRIMARY KEY,
                detection_id TEXT,
                sequence_id TEXT,
                product_id TEXT,
                product_name TEXT,
                product_batch TEXT,
                product_model TEXT,
                result TEXT,
                defect_types TEXT,
                defect_count INTEGER,
                inference_time_ms REAL,
                model_version TEXT,
                timestamp REAL,
                line_id TEXT,
                station_id TEXT,
                camera_id TEXT,
                original_image_path TEXT,
                annotated_image_path TEXT,
                thumbnail_path TEXT,
                defects_detail TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_product_id ON {table_name}(product_id)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_result ON {table_name}(result)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_defect_types ON {table_name}(defect_types)")
        self._conn.commit()

    def _get_table_name(self, timestamp: float) -> str:
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
        return f"detection_records_{date_str}"

    def save_record(self, record: DetectionRecord) -> bool:
        if not self._enabled or not self._conn:
            return False

        with self._lock:
            try:
                table_name = self._get_table_name(record.timestamp)
                self._ensure_table(table_name)

                cursor = self._conn.cursor()
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (record_id, detection_id, sequence_id, product_id, product_name,
                     product_batch, product_model, result, defect_types, defect_count,
                     inference_time_ms, model_version, timestamp, line_id, station_id,
                     camera_id, original_image_path, annotated_image_path, thumbnail_path,
                     defects_detail, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.detection_id,
                    record.sequence_id,
                    record.product_id,
                    record.product_name,
                    record.product_batch,
                    record.product_model,
                    record.result.value,
                    record.defect_types,
                    record.defect_count,
                    record.inference_time_ms,
                    record.model_version,
                    record.timestamp,
                    record.line_id,
                    record.station_id,
                    record.camera_id,
                    record.original_image_path,
                    record.annotated_image_path,
                    record.thumbnail_path,
                    record.defects_detail,
                    json.dumps(record.metadata, ensure_ascii=False) if isinstance(record.metadata, dict) else str(record.metadata)
                ))
                self._conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save record: {e}", exc_info=True)
                return False

    def query_records(self, product_id=None, start_time=None, end_time=None,
                      result=None, defect_type=None, limit=100, offset=0) -> Tuple[List[Dict], int]:
        if not self._enabled or not self._conn:
            return [], 0

        with self._lock:
            try:
                tables = self.get_tables()
                if not tables:
                    return [], 0

                conditions = []
                params = []

                if product_id:
                    conditions.append("product_id = ?")
                    params.append(product_id)
                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(start_time)
                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(end_time)
                if result:
                    conditions.append("result = ?")
                    params.append(result)
                if defect_type:
                    conditions.append("defect_types LIKE ?")
                    params.append(f"%{defect_type}%")

                where_clause = ""
                if conditions:
                    where_clause = " WHERE " + " AND ".join(conditions)

                union_queries = []
                count_queries = []
                for table in tables:
                    union_queries.append(f"SELECT * FROM {table}{where_clause}")
                    count_queries.append(f"SELECT COUNT(*) as cnt FROM {table}{where_clause}")

                full_query = " UNION ALL ".join(union_queries) + " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                count_query = " + ".join([f"({q})" for q in count_queries])

                count_params = params * len(tables)
                cursor = self._conn.cursor()
                cursor.execute(count_query, count_params)
                total_count = cursor.fetchone()[0]

                query_params = params * len(tables) + [limit, offset]
                cursor = self._conn.cursor()
                cursor.execute(full_query, query_params)
                rows = cursor.fetchall()

                records = []
                for row in rows:
                    record_dict = dict(row)
                    if record_dict.get("metadata"):
                        try:
                            record_dict["metadata"] = json.loads(record_dict["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    records.append(record_dict)

                return records, total_count
            except Exception as e:
                logger.error(f"Failed to query records: {e}", exc_info=True)
                return [], 0

    def get_record_by_detection_id(self, detection_id: str) -> Optional[Dict]:
        if not self._enabled or not self._conn:
            return None

        with self._lock:
            try:
                tables = self.get_tables()
                for table in tables:
                    cursor = self._conn.cursor()
                    cursor.execute(f"SELECT * FROM {table} WHERE detection_id = ?", (detection_id,))
                    row = cursor.fetchone()
                    if row:
                        record_dict = dict(row)
                        if record_dict.get("metadata"):
                            try:
                                record_dict["metadata"] = json.loads(record_dict["metadata"])
                            except (json.JSONDecodeError, TypeError):
                                pass
                        return record_dict
                return None
            except Exception as e:
                logger.error(f"Failed to get record by detection_id: {e}", exc_info=True)
                return None

    def get_tables(self) -> List[str]:
        if not self._enabled or not self._conn:
            return []

        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'detection_records_%' ORDER BY name"
                )
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get tables: {e}", exc_info=True)
                return []

    def cleanup_old_tables(self, retention_days: int) -> int:
        if not self._enabled or not self._conn:
            return 0

        with self._lock:
            try:
                tables = self.get_tables()
                cutoff = datetime.now()
                dropped = 0
                for table in tables:
                    try:
                        date_str = table.replace("detection_records_", "")
                        table_date = datetime.strptime(date_str, "%Y%m%d")
                        age_days = (cutoff - table_date).days
                        if age_days > retention_days:
                            cursor = self._conn.cursor()
                            cursor.execute(f"DROP TABLE IF EXISTS {table}")
                            dropped += 1
                            logger.info(f"Dropped old table: {table}")
                    except ValueError:
                        continue
                if dropped > 0:
                    self._conn.commit()
                return dropped
            except Exception as e:
                logger.error(f"Failed to cleanup old tables: {e}", exc_info=True)
                return 0

    def close(self):
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception as e:
                    logger.warning(f"Error closing SQLite connection: {e}")
                self._conn = None
        logger.info("Record store closed")
