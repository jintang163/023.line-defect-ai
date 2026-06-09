from typing import Dict, Any, Optional, List, Tuple
import threading
import time
import json
from datetime import datetime

from src.utils.schemas import DetectionRecord, DetectionResult
from src.utils.logger import Logger

logger = Logger("record_store", "INFO", "./logs/defect-detection.log").logger

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False
    logger.warning("psycopg2 not available, record store disabled")


class RecordStore:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)
        self._lock = threading.RLock()
        self._pool: Optional[Any] = None

        self._pg_host = config.get("host", "localhost")
        self._pg_port = config.get("port", 5432)
        self._pg_database = config.get("database", "defect_db")
        self._pg_user = config.get("user", "defect")
        self._pg_password = config.get("password", "defect123")
        self._pg_min_conn = config.get("min_connections", 2)
        self._pg_max_conn = config.get("max_connections", 10)

        if self._enabled and PG_AVAILABLE:
            self._init_db()

    def _get_conn(self):
        if self._pool:
            return self._pool.getconn()
        return None

    def _put_conn(self, conn, close=False):
        if self._pool and conn:
            self._pool.putconn(conn, close=close)

    def _init_db(self):
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self._pg_min_conn,
                maxconn=self._pg_max_conn,
                host=self._pg_host,
                port=self._pg_port,
                database=self._pg_database,
                user=self._pg_user,
                password=self._pg_password
            )
            logger.info(f"PostgreSQL connection pool created: {self._pg_host}:{self._pg_port}/{self._pg_database}")

            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS detection_records (
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
                        inference_time_ms DOUBLE PRECISION,
                        model_version TEXT,
                        timestamp DOUBLE PRECISION,
                        line_id TEXT,
                        station_id TEXT,
                        camera_id TEXT,
                        original_image_path TEXT,
                        annotated_image_path TEXT,
                        thumbnail_path TEXT,
                        defects_detail TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_product_id ON detection_records(product_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_timestamp ON detection_records(timestamp)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_result ON detection_records(result)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_defect_types ON detection_records(defect_types)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_product_model ON detection_records(product_model)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_detection_id ON detection_records(detection_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_created_at ON detection_records(created_at)")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alert_events (
                        alert_id TEXT PRIMARY KEY,
                        level TEXT,
                        category TEXT,
                        message TEXT,
                        source TEXT,
                        action TEXT,
                        grade TEXT,
                        timestamp DOUBLE PRECISION,
                        detection_id TEXT,
                        defect_id TEXT,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        acknowledged_by TEXT,
                        acknowledged_at DOUBLE PRECISION,
                        details JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ae_level ON alert_events(level)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ae_category ON alert_events(category)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ae_grade ON alert_events(grade)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ae_timestamp ON alert_events(timestamp)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ae_acknowledged ON alert_events(acknowledged)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ae_created_at ON alert_events(created_at)")
                conn.commit()
                cur.close()
                logger.info("PostgreSQL detection_records table ensured with indexes")
            finally:
                self._put_conn(conn)
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}", exc_info=True)
            self._enabled = False

    def _ensure_partition(self, date_str: str, conn):
        partition_name = f"detection_records_{date_str}"
        cur = conn.cursor()
        cur.execute("""
            SELECT 1 FROM pg_class WHERE relname = %s
        """, (partition_name,))
        if not cur.fetchone():
            start_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            next_day_sql = "SELECT (%s::date + INTERVAL '1 day')::date"
            cur.execute(next_day_sql, (start_date,))
            end_date = cur.fetchone()[0].isoformat()
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {partition} PARTITION OF detection_records
                FOR VALUES FROM (%s) TO (%s)
            """).format(partition=sql.Identifier(partition_name)), (start_date, end_date))
            conn.commit()
            logger.info(f"Created partition: {partition_name} [{start_date}, {end_date})")
        cur.close()

    def _ensure_standalone_partition_fallback(self, date_str: str, conn):
        partition_name = f"detection_records_{date_str}"
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_class WHERE relname = %s", (partition_name,))
        if not cur.fetchone():
            start_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {partition} (
                    LIKE detection_records INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES
                )
            """).format(partition=sql.Identifier(partition_name)))
            conn.commit()
            logger.info(f"Created standalone table (non-partitioned): {partition_name}")
        cur.close()

    def save_record(self, record: DetectionRecord) -> bool:
        if not self._enabled or not self._pool:
            return False

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return False
            try:
                date_str = datetime.fromtimestamp(record.timestamp).strftime("%Y%m%d")

                try:
                    self._ensure_partition(date_str, conn)
                except Exception as e:
                    logger.warning(f"Partition creation failed (may not be partitioned table), trying standalone: {e}")
                    conn.rollback()
                    try:
                        self._ensure_standalone_partition_fallback(date_str, conn)
                    except Exception as e2:
                        logger.warning(f"Standalone table creation also failed: {e2}")
                        conn.rollback()

                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO detection_records
                    (record_id, detection_id, sequence_id, product_id, product_name,
                     product_batch, product_model, result, defect_types, defect_count,
                     inference_time_ms, model_version, timestamp, line_id, station_id,
                     camera_id, original_image_path, annotated_image_path, thumbnail_path,
                     defects_detail, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (record_id) DO UPDATE SET
                        result = EXCLUDED.result,
                        defect_types = EXCLUDED.defect_types,
                        defect_count = EXCLUDED.defect_count,
                        defects_detail = EXCLUDED.defects_detail
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
                    json.dumps(record.metadata, ensure_ascii=False) if isinstance(record.metadata, dict) else "{}"
                ))
                conn.commit()
                cur.close()
                return True
            except Exception as e:
                logger.error(f"Failed to save record: {e}", exc_info=True)
                conn.rollback()
                return False
            finally:
                self._put_conn(conn)

    def query_records(self, product_id=None, product_model=None, start_time=None, end_time=None,
                      result=None, defect_type=None, limit=100, offset=0) -> Tuple[List[Dict], int]:
        if not self._enabled or not self._pool:
            return [], 0

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return [], 0
            try:
                conditions = []
                params = []

                if product_id:
                    conditions.append("product_id = %s")
                    params.append(product_id)
                if product_model:
                    conditions.append("product_model = %s")
                    params.append(product_model)
                if start_time:
                    conditions.append("timestamp >= %s")
                    params.append(start_time)
                if end_time:
                    conditions.append("timestamp <= %s")
                    params.append(end_time)
                if result:
                    conditions.append("result = %s")
                    params.append(result)
                if defect_type:
                    conditions.append("defect_types LIKE %s")
                    params.append(f"%{defect_type}%")

                where_clause = ""
                if conditions:
                    where_clause = " WHERE " + " AND ".join(conditions)

                count_query = f"SELECT COUNT(*) as cnt FROM detection_records{where_clause}"
                cur = conn.cursor()
                cur.execute(count_query, params)
                total_count = cur.fetchone()[0]
                cur.close()

                full_query = f"""
                    SELECT * FROM detection_records{where_clause}
                    ORDER BY timestamp DESC LIMIT %s OFFSET %s
                """
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(full_query, params + [limit, offset])
                rows = cur.fetchall()

                records = []
                for row in rows:
                    record_dict = dict(row)
                    if record_dict.get("metadata") and isinstance(record_dict["metadata"], str):
                        try:
                            record_dict["metadata"] = json.loads(record_dict["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if record_dict.get("created_at"):
                        record_dict["created_at"] = str(record_dict["created_at"])
                    records.append(record_dict)

                cur.close()
                return records, total_count
            except Exception as e:
                logger.error(f"Failed to query records: {e}", exc_info=True)
                return [], 0
            finally:
                self._put_conn(conn)

    def get_record_by_detection_id(self, detection_id: str) -> Optional[Dict]:
        if not self._enabled or not self._pool:
            return None

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return None
            try:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(
                    "SELECT * FROM detection_records WHERE detection_id = %s LIMIT 1",
                    (detection_id,)
                )
                row = cur.fetchone()
                cur.close()
                if row:
                    record_dict = dict(row)
                    if record_dict.get("metadata") and isinstance(record_dict["metadata"], str):
                        try:
                            record_dict["metadata"] = json.loads(record_dict["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if record_dict.get("created_at"):
                        record_dict["created_at"] = str(record_dict["created_at"])
                    return record_dict
                return None
            except Exception as e:
                logger.error(f"Failed to get record by detection_id: {e}", exc_info=True)
                return None
            finally:
                self._put_conn(conn)

    def get_tables(self) -> List[str]:
        if not self._enabled or not self._pool:
            return []

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return []
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE tablename LIKE 'detection_records_%'
                    AND schemaname = 'public'
                    ORDER BY tablename
                """)
                return [row[0] for row in cur.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get tables: {e}", exc_info=True)
                return []
            finally:
                self._put_conn(conn)

    def get_distinct_values(self, column: str) -> List[str]:
        if not self._enabled or not self._pool:
            return []

        allowed = {"product_id", "product_model", "defect_types", "result"}
        if column not in allowed:
            return []

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return []
            try:
                cur = conn.cursor()
                cur.execute(
                    sql.SQL("SELECT DISTINCT {col} FROM detection_records WHERE {col} IS NOT NULL AND {col} != '' ORDER BY {col}").format(
                        col=sql.Identifier(column)
                    )
                )
                return [row[0] for row in cur.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get distinct values for {column}: {e}", exc_info=True)
                return []
            finally:
                self._put_conn(conn)

    def save_alert_event(self, alert_data: Dict[str, Any]) -> bool:
        if not self._enabled or not self._pool:
            return False

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return False
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO alert_events
                    (alert_id, level, category, message, source, action, grade,
                     timestamp, detection_id, defect_id, acknowledged, acknowledged_by,
                     acknowledged_at, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (alert_id) DO UPDATE SET
                        acknowledged = EXCLUDED.acknowledged,
                        acknowledged_by = EXCLUDED.acknowledged_by,
                        acknowledged_at = EXCLUDED.acknowledged_at
                """, (
                    alert_data.get("alert_id", ""),
                    alert_data.get("level", ""),
                    alert_data.get("category", ""),
                    alert_data.get("message", ""),
                    alert_data.get("source", ""),
                    alert_data.get("action", ""),
                    alert_data.get("grade", ""),
                    alert_data.get("timestamp", 0.0),
                    alert_data.get("detection_id", ""),
                    alert_data.get("defect_id", ""),
                    alert_data.get("acknowledged", False),
                    alert_data.get("acknowledged_by"),
                    alert_data.get("acknowledged_at"),
                    json.dumps(alert_data.get("details", {}), ensure_ascii=False, default=str)
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save alert event: {e}", exc_info=True)
                conn.rollback()
                return False
            finally:
                self._put_conn(conn)

    def query_alert_events(self, level: Optional[str] = None,
                           category: Optional[str] = None,
                           grade: Optional[str] = None,
                           acknowledged: Optional[bool] = None,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           limit: int = 100,
                           offset: int = 0) -> List[Dict[str, Any]]:
        if not self._enabled or not self._pool:
            return []

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return []
            try:
                conditions = []
                params = []
                if level:
                    conditions.append("level = %s")
                    params.append(level)
                if category:
                    conditions.append("category = %s")
                    params.append(category)
                if grade:
                    conditions.append("grade = %s")
                    params.append(grade)
                if acknowledged is not None:
                    conditions.append("acknowledged = %s")
                    params.append(acknowledged)
                if start_time:
                    conditions.append("timestamp >= %s")
                    params.append(start_time)
                if end_time:
                    conditions.append("timestamp <= %s")
                    params.append(end_time)

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                params.extend([limit, offset])

                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(f"""
                    SELECT * FROM alert_events
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                """, params)
                rows = cur.fetchall()
                for row in rows:
                    if isinstance(row.get("details"), str):
                        try:
                            row["details"] = json.loads(row["details"])
                        except:
                            pass
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to query alert events: {e}", exc_info=True)
                return []
            finally:
                self._put_conn(conn)

    def update_alert_acknowledged(self, alert_id: str, operator: str) -> bool:
        if not self._enabled or not self._pool:
            return False

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return False
            try:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE alert_events
                    SET acknowledged = TRUE, acknowledged_by = %s, acknowledged_at = %s
                    WHERE alert_id = %s
                """, (operator, time.time(), alert_id))
                conn.commit()
                return cur.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to update alert acknowledged: {e}", exc_info=True)
                conn.rollback()
                return False
            finally:
                self._put_conn(conn)

    def count_alert_events(self, level: Optional[str] = None,
                           category: Optional[str] = None,
                           grade: Optional[str] = None,
                           acknowledged: Optional[bool] = None,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> int:
        if not self._enabled or not self._pool:
            return 0

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return 0
            try:
                conditions = []
                params = []
                if level:
                    conditions.append("level = %s")
                    params.append(level)
                if category:
                    conditions.append("category = %s")
                    params.append(category)
                if grade:
                    conditions.append("grade = %s")
                    params.append(grade)
                if acknowledged is not None:
                    conditions.append("acknowledged = %s")
                    params.append(acknowledged)
                if start_time:
                    conditions.append("timestamp >= %s")
                    params.append(start_time)
                if end_time:
                    conditions.append("timestamp <= %s")
                    params.append(end_time)

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM alert_events WHERE {where_clause}", params)
                return cur.fetchone()[0]
            except Exception as e:
                logger.error(f"Failed to count alert events: {e}", exc_info=True)
                return 0
            finally:
                self._put_conn(conn)

    def cleanup_old_tables(self, retention_days: int) -> int:
        if not self._enabled or not self._pool:
            return 0

        with self._lock:
            conn = self._get_conn()
            if not conn:
                return 0
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
                            cur = conn.cursor()
                            cur.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(
                                table=sql.Identifier(table)
                            ))
                            conn.commit()
                            cur.close()
                            dropped += 1
                            logger.info(f"Dropped old partition/table: {table}")
                    except ValueError:
                        continue
                return dropped
            except Exception as e:
                logger.error(f"Failed to cleanup old tables: {e}", exc_info=True)
                conn.rollback()
                return 0
            finally:
                self._put_conn(conn)

    def close(self):
        with self._lock:
            if self._pool:
                try:
                    self._pool.closeall()
                except Exception as e:
                    logger.warning(f"Error closing PostgreSQL pool: {e}")
                self._pool = None
        logger.info("Record store closed (PostgreSQL)")
