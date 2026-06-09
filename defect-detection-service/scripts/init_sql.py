import os
import sys
import psycopg2
from psycopg2 import sql

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import ConfigManager
from src.utils.logger import Logger

logger = Logger("init_sql", "INFO", "./logs/defect-detection.log").logger


def init_postgres(config_path: str = "./config/config.yaml"):
    config_manager = ConfigManager(config_path)
    dm_config = config_manager.get_data_management_config()
    record_config = dm_config.get("record_storage", {})

    host = record_config.get("host", "localhost")
    port = record_config.get("port", 5432)
    database = record_config.get("database", "defect_db")
    user = record_config.get("user", "defect")
    password = record_config.get("password", "defect123")

    logger.info(f"Connecting to PostgreSQL: {host}:{port}/{database}")

    try:
        conn = psycopg2.connect(
            host=host, port=port, database=database,
            user=user, password=password
        )
        conn.autocommit = True
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
            ) PARTITION BY RANGE (created_at)
        """)
        logger.info("Created detection_records as partitioned table (PARTITION BY RANGE)")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_product_id ON detection_records(product_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_timestamp ON detection_records(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_result ON detection_records(result)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_defect_types ON detection_records(defect_types)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_product_model ON detection_records(product_model)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_detection_id ON detection_records(detection_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dr_created_at ON detection_records(created_at)")
        logger.info("Created indexes on detection_records")

        from datetime import datetime, timedelta
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(-7, 31):
            day = today + timedelta(days=i)
            next_day = day + timedelta(days=1)
            partition_name = f"detection_records_{day.strftime('%Y%m%d')}"
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {partition} PARTITION OF detection_records
                FOR VALUES FROM (%s) TO (%s)
            """).format(partition=sql.Identifier(partition_name)), (day, next_day))
        logger.info("Created date-range partitions for -7 to +30 days")

        cur.execute("""
            SELECT tablename FROM pg_tables
            WHERE tablename LIKE 'detection_records_%' AND schemaname = 'public'
            ORDER BY tablename
        """)
        partitions = [row[0] for row in cur.fetchall()]
        logger.info(f"Current partitions: {partitions}")

        cur.close()
        conn.close()
        logger.info("PostgreSQL schema initialization completed successfully")

    except psycopg2.errors.InvalidTableDefinition:
        logger.warning("PARTITION BY RANGE not supported or table already exists as non-partitioned, creating as regular table")
        try:
            conn.rollback()
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
            conn.commit()
            cur.close()
            logger.info("Created detection_records as regular (non-partitioned) table")
        except Exception as e2:
            logger.error(f"Failed to create fallback table: {e2}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}", exc_info=True)


if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH", "./config/config.yaml")
    init_postgres(config_path)
