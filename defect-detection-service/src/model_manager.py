import os
import json
import time
import shutil
import threading
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from src.utils.schemas import (
    ModelVersion, ModelVersionStatus, AlgorithmType,
    AnnotationRecord, AnnotationType, DefectType,
    ABTestConfig, ABTestStatus,
    RetrainTrigger, RetrainTriggerStatus
)
from src.utils.logger import Logger

logger = Logger("model-management", "INFO", "./logs/model-management.log").logger

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)
        self._models_dir = config.get("models_dir", "./models")
        self._versions_db_path = config.get("versions_db_path", "./data/model_versions.db")
        self._annotations_db_path = config.get("annotations_db_path", "./data/annotations.db")
        self._collection_dir = config.get("collection_dir", "./data/retrain_samples")

        self._retrain_false_positive_threshold = config.get("retrain_false_positive_threshold", 0.05)
        self._retrain_false_negative_threshold = config.get("retrain_false_negative_threshold", 0.05)
        self._retrain_consecutive_days = config.get("retrain_consecutive_days", 7)
        self._retrain_sample_days = config.get("retrain_sample_days", 14)
        self._retrain_auto_trigger = config.get("retrain_auto_trigger", True)
        self._retrain_command = config.get("retrain_command", "")
        self._retrain_check_interval_sec = config.get("retrain_check_interval_sec", 3600)

        self._lock = threading.RLock()
        self._versions: Dict[str, ModelVersion] = {}
        self._annotations: List[AnnotationRecord] = []
        self._ab_tests: Dict[str, ABTestConfig] = {}
        self._retrain_triggers: List[RetrainTrigger] = []

        self._false_positive_daily: Dict[str, List[float]] = {}
        self._false_negative_daily: Dict[str, List[float]] = {}

        self._versions_db_conn: Optional[sqlite3.Connection] = None
        self._annotations_db_conn: Optional[sqlite3.Connection] = None

        self._retrain_check_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        if self._enabled:
            self._init_databases()
            self._load_versions_from_db()
            self._load_annotations_from_db()
            self._load_ab_tests_from_db()
            self._load_retrain_triggers_from_db()
            self._start_retrain_checker()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _init_databases(self):
        for db_path, attr_name in [
            (self._versions_db_path, "_versions_db_conn"),
            (self._annotations_db_path, "_annotations_db_conn")
        ]:
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                setattr(self, attr_name, conn)
            except Exception as e:
                logger.error(f"Failed to init database {db_path}: {e}")

        self._init_versions_db_schema()
        self._init_annotations_db_schema()
        self._init_ab_tests_db_schema()
        self._init_retrain_triggers_db_schema()

    def _init_versions_db_schema(self):
        if not self._versions_db_conn:
            return
        cursor = self._versions_db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version_tag TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_mb REAL NOT NULL,
                algorithm_type TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT DEFAULT '',
                uploaded_by TEXT DEFAULT 'system',
                created_at REAL NOT NULL,
                metrics TEXT DEFAULT '{}',
                canary_lines TEXT DEFAULT '[]',
                canary_traffic_percent REAL DEFAULT 0.0,
                parent_version_id TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                file_hash TEXT DEFAULT ''
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_model_name ON model_versions(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_status ON model_versions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_algorithm_type ON model_versions(algorithm_type)")
        self._versions_db_conn.commit()

    def _init_annotations_db_schema(self):
        if not self._annotations_db_conn:
            return
        cursor = self._annotations_db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY,
                detection_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                annotation_type TEXT NOT NULL,
                original_defect_type TEXT DEFAULT '',
                corrected_defect_type TEXT DEFAULT '',
                original_bbox TEXT DEFAULT '{}',
                corrected_bbox TEXT DEFAULT '{}',
                annotator TEXT DEFAULT 'admin',
                timestamp REAL NOT NULL,
                notes TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_detection_id ON annotations(detection_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_type ON annotations(annotation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_annotator ON annotations(annotator)")
        self._annotations_db_conn.commit()

    def _init_ab_tests_db_schema(self):
        if not self._versions_db_conn:
            return
        cursor = self._versions_db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_a_version_id TEXT NOT NULL,
                model_b_version_id TEXT NOT NULL,
                status TEXT NOT NULL,
                traffic_split_percent REAL DEFAULT 50.0,
                target_lines TEXT DEFAULT '[]',
                started_at REAL DEFAULT 0.0,
                ended_at REAL DEFAULT 0.0,
                metrics_a TEXT DEFAULT '{}',
                metrics_b TEXT DEFAULT '{}',
                sample_count_a INTEGER DEFAULT 0,
                sample_count_b INTEGER DEFAULT 0,
                created_by TEXT DEFAULT 'admin',
                min_sample_size INTEGER DEFAULT 1000,
                confidence_level REAL DEFAULT 0.95,
                winner TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._versions_db_conn.commit()

    def _init_retrain_triggers_db_schema(self):
        if not self._versions_db_conn:
            return
        cursor = self._versions_db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrain_triggers (
                trigger_id TEXT PRIMARY KEY,
                trigger_reason TEXT NOT NULL,
                status TEXT NOT NULL,
                threshold_type TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                actual_value REAL NOT NULL,
                consecutive_days INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                algorithm_type TEXT NOT NULL,
                sample_count INTEGER DEFAULT 0,
                started_at REAL NOT NULL,
                completed_at REAL DEFAULT 0.0,
                new_model_version_id TEXT DEFAULT '',
                collection_dir TEXT DEFAULT '',
                training_log TEXT DEFAULT '',
                metrics TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._versions_db_conn.commit()

    def _load_versions_from_db(self):
        if not self._versions_db_conn:
            return
        try:
            cursor = self._versions_db_conn.cursor()
            cursor.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
            for row in cursor.fetchall():
                row_dict = dict(row)
                version = ModelVersion(
                    version_id=row_dict["version_id"],
                    model_name=row_dict["model_name"],
                    version_tag=row_dict["version_tag"],
                    file_path=row_dict["file_path"],
                    file_size_mb=row_dict["file_size_mb"],
                    algorithm_type=AlgorithmType(row_dict["algorithm_type"]),
                    status=ModelVersionStatus(row_dict["status"]),
                    description=row_dict.get("description", ""),
                    uploaded_by=row_dict.get("uploaded_by", "system"),
                    created_at=row_dict["created_at"],
                    metrics=json.loads(row_dict.get("metrics", "{}")),
                    canary_lines=json.loads(row_dict.get("canary_lines", "[]")),
                    canary_traffic_percent=row_dict.get("canary_traffic_percent", 0.0),
                    parent_version_id=row_dict.get("parent_version_id", ""),
                    metadata=json.loads(row_dict.get("metadata", "{}"))
                )
                self._versions[version.version_id] = version
            logger.info(f"Loaded {len(self._versions)} model versions from DB")
        except Exception as e:
            logger.error(f"Failed to load model versions: {e}")

    def _load_annotations_from_db(self):
        if not self._annotations_db_conn:
            return
        try:
            cursor = self._annotations_db_conn.cursor()
            cursor.execute("SELECT * FROM annotations ORDER BY timestamp DESC LIMIT 10000")
            for row in cursor.fetchall():
                row_dict = dict(row)
                annotation = AnnotationRecord(
                    annotation_id=row_dict["annotation_id"],
                    detection_id=row_dict["detection_id"],
                    image_path=row_dict["image_path"],
                    annotation_type=AnnotationType(row_dict["annotation_type"]),
                    original_defect_type=row_dict.get("original_defect_type", ""),
                    corrected_defect_type=row_dict.get("corrected_defect_type", ""),
                    original_bbox=json.loads(row_dict.get("original_bbox", "{}")),
                    corrected_bbox=json.loads(row_dict.get("corrected_bbox", "{}")),
                    annotator=row_dict.get("annotator", "admin"),
                    timestamp=row_dict["timestamp"],
                    notes=row_dict.get("notes", ""),
                    metadata=json.loads(row_dict.get("metadata", "{}"))
                )
                self._annotations.append(annotation)
            logger.info(f"Loaded {len(self._annotations)} annotations from DB")
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")

    def _load_ab_tests_from_db(self):
        if not self._versions_db_conn:
            return
        try:
            cursor = self._versions_db_conn.cursor()
            cursor.execute("SELECT * FROM ab_tests ORDER BY started_at DESC")
            for row in cursor.fetchall():
                row_dict = dict(row)
                test = ABTestConfig(
                    test_id=row_dict["test_id"],
                    name=row_dict["name"],
                    model_a_version_id=row_dict["model_a_version_id"],
                    model_b_version_id=row_dict["model_b_version_id"],
                    status=ABTestStatus(row_dict["status"]),
                    traffic_split_percent=row_dict.get("traffic_split_percent", 50.0),
                    target_lines=json.loads(row_dict.get("target_lines", "[]")),
                    started_at=row_dict.get("started_at", 0.0),
                    ended_at=row_dict.get("ended_at", 0.0),
                    metrics_a=json.loads(row_dict.get("metrics_a", "{}")),
                    metrics_b=json.loads(row_dict.get("metrics_b", "{}")),
                    sample_count_a=row_dict.get("sample_count_a", 0),
                    sample_count_b=row_dict.get("sample_count_b", 0),
                    created_by=row_dict.get("created_by", "admin"),
                    min_sample_size=row_dict.get("min_sample_size", 1000),
                    confidence_level=row_dict.get("confidence_level", 0.95),
                    winner=row_dict.get("winner", ""),
                    notes=row_dict.get("notes", ""),
                    metadata=json.loads(row_dict.get("metadata", "{}"))
                )
                self._ab_tests[test.test_id] = test
            logger.info(f"Loaded {len(self._ab_tests)} A/B tests from DB")
        except Exception as e:
            logger.error(f"Failed to load A/B tests: {e}")

    def _load_retrain_triggers_from_db(self):
        if not self._versions_db_conn:
            return
        try:
            cursor = self._versions_db_conn.cursor()
            cursor.execute("SELECT * FROM retrain_triggers ORDER BY started_at DESC")
            for row in cursor.fetchall():
                row_dict = dict(row)
                trigger = RetrainTrigger(
                    trigger_id=row_dict["trigger_id"],
                    trigger_reason=row_dict["trigger_reason"],
                    status=RetrainTriggerStatus(row_dict["status"]),
                    threshold_type=row_dict["threshold_type"],
                    threshold_value=row_dict["threshold_value"],
                    actual_value=row_dict["actual_value"],
                    consecutive_days=row_dict["consecutive_days"],
                    product_id=row_dict["product_id"],
                    algorithm_type=AlgorithmType(row_dict["algorithm_type"]),
                    sample_count=row_dict.get("sample_count", 0),
                    started_at=row_dict["started_at"],
                    completed_at=row_dict.get("completed_at", 0.0),
                    new_model_version_id=row_dict.get("new_model_version_id", ""),
                    collection_dir=row_dict.get("collection_dir", ""),
                    training_log=row_dict.get("training_log", ""),
                    metrics=json.loads(row_dict.get("metrics", "{}")),
                    metadata=json.loads(row_dict.get("metadata", "{}"))
                )
                self._retrain_triggers.append(trigger)
            logger.info(f"Loaded {len(self._retrain_triggers)} retrain triggers from DB")
        except Exception as e:
            logger.error(f"Failed to load retrain triggers: {e}")

    def upload_model(self, model_name: str, version_tag: str, source_file_path: str,
                     algorithm_type: AlgorithmType, uploaded_by: str = "system",
                     description: str = "", parent_version_id: str = "",
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[ModelVersion]:
        with self._lock:
            try:
                if not os.path.exists(source_file_path):
                    logger.error(f"Model file not found: {source_file_path}")
                    return None

                file_size_mb = os.path.getsize(source_file_path) / (1024 * 1024)
                if file_size_mb < 0.1:
                    logger.error(f"Model file too small ({file_size_mb:.2f}MB), possibly corrupt")
                    return None

                file_hash = self._compute_file_hash(source_file_path)

                model_subdir = os.path.join(self._models_dir, model_name)
                os.makedirs(model_subdir, exist_ok=True)

                ext = os.path.splitext(source_file_path)[1]
                dest_filename = f"{model_name}_{version_tag}{ext}"
                dest_path = os.path.join(model_subdir, dest_filename)

                if os.path.exists(dest_path):
                    timestamp_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
                    dest_filename = f"{model_name}_{version_tag}_{timestamp_suffix}{ext}"
                    dest_path = os.path.join(model_subdir, dest_filename)

                shutil.copy2(source_file_path, dest_path)
                logger.info(f"Model file copied to: {dest_path}")

                version = ModelVersion.create(
                    model_name=model_name,
                    version_tag=version_tag,
                    file_path=dest_path,
                    file_size_mb=file_size_mb,
                    algorithm_type=algorithm_type,
                    uploaded_by=uploaded_by,
                    description=description,
                    parent_version_id=parent_version_id,
                    metadata=metadata or {}
                )

                self._versions[version.version_id] = version
                self._save_version_to_db(version, file_hash)
                logger.info(f"Model uploaded: {model_name} v{version_tag} ({version.version_id})")
                return version

            except Exception as e:
                logger.error(f"Failed to upload model: {e}", exc_info=True)
                return None

    def _compute_file_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_version_to_db(self, version: ModelVersion, file_hash: str = ""):
        if not self._versions_db_conn:
            return
        try:
            cursor = self._versions_db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO model_versions
                (version_id, model_name, version_tag, file_path, file_size_mb,
                 algorithm_type, status, description, uploaded_by, created_at,
                 metrics, canary_lines, canary_traffic_percent, parent_version_id,
                 metadata, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id, version.model_name, version.version_tag,
                version.file_path, version.file_size_mb,
                version.algorithm_type.value, version.status.value,
                version.description, version.uploaded_by, version.created_at,
                json.dumps(version.metrics, ensure_ascii=False),
                json.dumps(version.canary_lines, ensure_ascii=False),
                version.canary_traffic_percent, version.parent_version_id,
                json.dumps(version.metadata, ensure_ascii=False), file_hash
            ))
            self._versions_db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to save version to DB: {e}")

    def list_versions(self, model_name: str = None,
                      algorithm_type: AlgorithmType = None,
                      status: ModelVersionStatus = None) -> List[ModelVersion]:
        with self._lock:
            versions = list(self._versions.values())
            if model_name:
                versions = [v for v in versions if v.model_name == model_name]
            if algorithm_type:
                versions = [v for v in versions if v.algorithm_type == algorithm_type]
            if status:
                versions = [v for v in versions if v.status == status]
            versions.sort(key=lambda v: v.created_at, reverse=True)
            return versions

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        return self._versions.get(version_id)

    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        with self._lock:
            for v in self._versions.values():
                if v.model_name == model_name and v.status == ModelVersionStatus.PRODUCTION:
                    return v
            return None

    def rollback_to_version(self, version_id: str, operator: str = "system") -> bool:
        with self._lock:
            target = self._versions.get(version_id)
            if not target:
                logger.error(f"Version not found: {version_id}")
                return False

            current_prod = None
            for v in self._versions.values():
                if v.model_name == target.model_name and v.status == ModelVersionStatus.PRODUCTION:
                    current_prod = v
                    break

            if current_prod:
                current_prod.status = ModelVersionStatus.DEPRECATED
                self._save_version_to_db(current_prod)
                logger.info(f"Deprecated production version: {current_prod.version_tag}")

            target.status = ModelVersionStatus.PRODUCTION
            target.canary_lines = []
            target.canary_traffic_percent = 0.0
            self._save_version_to_db(target)
            logger.info(f"Rolled back to version: {target.version_tag} by {operator}")
            return True

    def promote_to_canary(self, version_id: str, canary_lines: List[str],
                          traffic_percent: float = 10.0) -> bool:
        with self._lock:
            version = self._versions.get(version_id)
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False

            if version.status not in (ModelVersionStatus.DRAFT, ModelVersionStatus.STAGING):
                logger.error(f"Version {version_id} status {version.status.value} cannot be promoted to canary")
                return False

            version.status = ModelVersionStatus.CANARY
            version.canary_lines = canary_lines
            version.canary_traffic_percent = traffic_percent
            self._save_version_to_db(version)
            logger.info(f"Version {version.version_tag} promoted to canary on lines: {canary_lines}")
            return True

    def promote_to_production(self, version_id: str) -> bool:
        with self._lock:
            version = self._versions.get(version_id)
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False

            if version.status not in (ModelVersionStatus.CANARY, ModelVersionStatus.STAGING):
                logger.error(f"Version {version_id} status {version.status.value} cannot be promoted to production")
                return False

            current_prod = None
            for v in self._versions.values():
                if v.model_name == version.model_name and v.status == ModelVersionStatus.PRODUCTION:
                    current_prod = v
                    break

            if current_prod:
                current_prod.status = ModelVersionStatus.DEPRECATED
                self._save_version_to_db(current_prod)

            version.status = ModelVersionStatus.PRODUCTION
            version.canary_lines = []
            version.canary_traffic_percent = 0.0
            self._save_version_to_db(version)
            logger.info(f"Version {version.version_tag} promoted to production")
            return True

    def update_version_metrics(self, version_id: str, metrics: Dict[str, Any]) -> bool:
        with self._lock:
            version = self._versions.get(version_id)
            if not version:
                return False
            version.metrics.update(metrics)
            self._save_version_to_db(version)
            return True

    def get_canary_version_for_line(self, model_name: str, line_id: str) -> Optional[ModelVersion]:
        with self._lock:
            for v in self._versions.values():
                if v.model_name == model_name and v.status == ModelVersionStatus.CANARY:
                    if line_id in v.canary_lines:
                        return v
            return None

    def should_use_canary(self, model_name: str, line_id: str) -> bool:
        canary = self.get_canary_version_for_line(model_name, line_id)
        if not canary:
            return False
        import random
        return random.random() * 100 < canary.canary_traffic_percent

    def create_annotation(self, detection_id: str, image_path: str,
                          annotation_type: AnnotationType,
                          original_defect_type: str = "",
                          corrected_defect_type: str = "",
                          original_bbox: Optional[Dict[str, float]] = None,
                          corrected_bbox: Optional[Dict[str, float]] = None,
                          annotator: str = "admin",
                          notes: str = "",
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[AnnotationRecord]:
        with self._lock:
            try:
                annotation = AnnotationRecord.create(
                    detection_id=detection_id,
                    image_path=image_path,
                    annotation_type=annotation_type,
                    original_defect_type=original_defect_type,
                    corrected_defect_type=corrected_defect_type,
                    original_bbox=original_bbox,
                    corrected_bbox=corrected_bbox,
                    annotator=annotator,
                    notes=notes,
                    metadata=metadata
                )
                self._annotations.append(annotation)
                self._save_annotation_to_db(annotation)

                self._update_error_rate(annotation)

                logger.info(f"Annotation created: {annotation.annotation_id} for detection {detection_id}")
                return annotation

            except Exception as e:
                logger.error(f"Failed to create annotation: {e}", exc_info=True)
                return None

    def _save_annotation_to_db(self, annotation: AnnotationRecord):
        if not self._annotations_db_conn:
            return
        try:
            cursor = self._annotations_db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO annotations
                (annotation_id, detection_id, image_path, annotation_type,
                 original_defect_type, corrected_defect_type, original_bbox,
                 corrected_bbox, annotator, timestamp, notes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                annotation.annotation_id, annotation.detection_id,
                annotation.image_path, annotation.annotation_type.value,
                annotation.original_defect_type, annotation.corrected_defect_type,
                json.dumps(annotation.original_bbox),
                json.dumps(annotation.corrected_bbox),
                annotation.annotator, annotation.timestamp,
                annotation.notes, json.dumps(annotation.metadata, ensure_ascii=False)
            ))
            self._annotations_db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to save annotation to DB: {e}")

    def _update_error_rate(self, annotation: AnnotationRecord):
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{annotation.annotation_type.value}_{today}"

        if annotation.annotation_type in (AnnotationType.FALSE_POSITIVE, AnnotationType.FALSE_NEGATIVE):
            daily_data = self._false_positive_daily if annotation.annotation_type == AnnotationType.FALSE_POSITIVE else self._false_negative_daily
            if key not in daily_data:
                daily_data[key] = []
            daily_data[key].append(time.time())

    def list_annotations(self, detection_id: str = None,
                         annotation_type: AnnotationType = None,
                         annotator: str = None,
                         limit: int = 100,
                         offset: int = 0) -> Tuple[List[AnnotationRecord], int]:
        with self._lock:
            results = list(self._annotations)
            if detection_id:
                results = [a for a in results if a.detection_id == detection_id]
            if annotation_type:
                results = [a for a in results if a.annotation_type == annotation_type]
            if annotator:
                results = [a for a in results if a.annotator == annotator]
            total = len(results)
            results.sort(key=lambda a: a.timestamp, reverse=True)
            return results[offset:offset + limit], total

    def create_ab_test(self, name: str, model_a_version_id: str,
                       model_b_version_id: str,
                       traffic_split_percent: float = 50.0,
                       target_lines: Optional[List[str]] = None,
                       created_by: str = "admin",
                       min_sample_size: int = 1000,
                       confidence_level: float = 0.95,
                       notes: str = "") -> Optional[ABTestConfig]:
        with self._lock:
            try:
                version_a = self._versions.get(model_a_version_id)
                version_b = self._versions.get(model_b_version_id)
                if not version_a or not version_b:
                    logger.error("One or both model versions not found for A/B test")
                    return None

                if version_a.algorithm_type != version_b.algorithm_type:
                    logger.error("A/B test versions must have the same algorithm type")
                    return None

                for test in self._ab_tests.values():
                    if test.status == ABTestStatus.RUNNING:
                        if (test.model_a_version_id in (model_a_version_id, model_b_version_id) or
                                test.model_b_version_id in (model_a_version_id, model_b_version_id)):
                            logger.error("One of the versions is already in a running A/B test")
                            return None

                test = ABTestConfig.create(
                    name=name,
                    model_a_version_id=model_a_version_id,
                    model_b_version_id=model_b_version_id,
                    traffic_split_percent=traffic_split_percent,
                    target_lines=target_lines,
                    created_by=created_by,
                    min_sample_size=min_sample_size,
                    confidence_level=confidence_level,
                    notes=notes
                )

                self._ab_tests[test.test_id] = test
                self._save_ab_test_to_db(test)
                logger.info(f"A/B test created: {name} ({test.test_id})")
                return test

            except Exception as e:
                logger.error(f"Failed to create A/B test: {e}", exc_info=True)
                return None

    def start_ab_test(self, test_id: str) -> bool:
        with self._lock:
            test = self._ab_tests.get(test_id)
            if not test:
                logger.error(f"A/B test not found: {test_id}")
                return False

            if test.status != ABTestStatus.PENDING:
                logger.error(f"A/B test {test_id} is not in PENDING status")
                return False

            version_b = self._versions.get(test.model_b_version_id)
            if version_b:
                version_b.status = ModelVersionStatus.STAGING
                self._save_version_to_db(version_b)

            test.status = ABTestStatus.RUNNING
            test.started_at = time.time()
            self._save_ab_test_to_db(test)
            logger.info(f"A/B test started: {test.name}")
            return True

    def stop_ab_test(self, test_id: str, winner: str = "") -> bool:
        with self._lock:
            test = self._ab_tests.get(test_id)
            if not test:
                return False

            if test.status != ABTestStatus.RUNNING:
                return False

            test.status = ABTestStatus.COMPLETED
            test.ended_at = time.time()
            test.winner = winner
            self._save_ab_test_to_db(test)

            if winner == "b":
                self.promote_to_production(test.model_b_version_id)
            elif winner == "a":
                version_b = self._versions.get(test.model_b_version_id)
                if version_b and version_b.status == ModelVersionStatus.STAGING:
                    version_b.status = ModelVersionStatus.DRAFT
                    self._save_version_to_db(version_b)

            logger.info(f"A/B test completed: {test.name}, winner: {winner}")
            return True

    def update_ab_test_metrics(self, test_id: str, model_label: str,
                               metrics: Dict[str, Any], sample_increment: int = 1) -> bool:
        with self._lock:
            test = self._ab_tests.get(test_id)
            if not test:
                return False

            if model_label == "a":
                test.metrics_a.update(metrics)
                test.sample_count_a += sample_increment
            elif model_label == "b":
                test.metrics_b.update(metrics)
                test.sample_count_b += sample_increment
            else:
                return False

            self._save_ab_test_to_db(test)
            return True

    def get_ab_test_for_inference(self, model_name: str, line_id: str = "") -> Optional[Tuple[str, str]]:
        with self._lock:
            for test in self._ab_tests.values():
                if test.status != ABTestStatus.RUNNING:
                    continue
                version_a = self._versions.get(test.model_a_version_id)
                version_b = self._versions.get(test.model_b_version_id)
                if not version_a or not version_b:
                    continue
                if version_a.model_name != model_name:
                    continue
                if test.target_lines and line_id not in test.target_lines:
                    continue
                import random
                if random.random() * 100 < test.traffic_split_percent:
                    return test.model_b_version_id, test.test_id
                else:
                    return test.model_a_version_id, test.test_id
            return None

    def list_ab_tests(self, status: ABTestStatus = None) -> List[ABTestConfig]:
        with self._lock:
            tests = list(self._ab_tests.values())
            if status:
                tests = [t for t in tests if t.status == status]
            tests.sort(key=lambda t: t.started_at if t.started_at > 0 else t.test_id, reverse=True)
            return tests

    def get_ab_test(self, test_id: str) -> Optional[ABTestConfig]:
        return self._ab_tests.get(test_id)

    def switch_full_to_version(self, test_id: str) -> bool:
        with self._lock:
            test = self._ab_tests.get(test_id)
            if not test or test.status != ABTestStatus.RUNNING:
                return False

            version_b = self._versions.get(test.model_b_version_id)
            if not version_b:
                return False

            test.status = ABTestStatus.COMPLETED
            test.ended_at = time.time()
            test.winner = "b"
            self._save_ab_test_to_db(test)

            self.promote_to_production(test.model_b_version_id)
            logger.info(f"Full switch to model B for A/B test {test.name}")
            return True

    def _save_ab_test_to_db(self, test: ABTestConfig):
        if not self._versions_db_conn:
            return
        try:
            cursor = self._versions_db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ab_tests
                (test_id, name, model_a_version_id, model_b_version_id, status,
                 traffic_split_percent, target_lines, started_at, ended_at,
                 metrics_a, metrics_b, sample_count_a, sample_count_b,
                 created_by, min_sample_size, confidence_level, winner, notes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test.test_id, test.name, test.model_a_version_id,
                test.model_b_version_id, test.status.value,
                test.traffic_split_percent,
                json.dumps(test.target_lines, ensure_ascii=False),
                test.started_at, test.ended_at,
                json.dumps(test.metrics_a, ensure_ascii=False),
                json.dumps(test.metrics_b, ensure_ascii=False),
                test.sample_count_a, test.sample_count_b,
                test.created_by, test.min_sample_size,
                test.confidence_level, test.winner, test.notes,
                json.dumps(test.metadata, ensure_ascii=False)
            ))
            self._versions_db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to save A/B test to DB: {e}")

    def _save_retrain_trigger_to_db(self, trigger: RetrainTrigger):
        if not self._versions_db_conn:
            return
        try:
            cursor = self._versions_db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO retrain_triggers
                (trigger_id, trigger_reason, status, threshold_type, threshold_value,
                 actual_value, consecutive_days, product_id, algorithm_type,
                 sample_count, started_at, completed_at, new_model_version_id,
                 collection_dir, training_log, metrics, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trigger.trigger_id, trigger.trigger_reason, trigger.status.value,
                trigger.threshold_type, trigger.threshold_value,
                trigger.actual_value, trigger.consecutive_days,
                trigger.product_id, trigger.algorithm_type.value,
                trigger.sample_count, trigger.started_at, trigger.completed_at,
                trigger.new_model_version_id, trigger.collection_dir,
                trigger.training_log,
                json.dumps(trigger.metrics, ensure_ascii=False),
                json.dumps(trigger.metadata, ensure_ascii=False)
            ))
            self._versions_db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to save retrain trigger to DB: {e}")

    def check_and_trigger_retrain(self, product_id: str = None,
                                  algorithm_type: AlgorithmType = None) -> Optional[RetrainTrigger]:
        with self._lock:
            try:
                fp_rates = self._compute_consecutive_daily_rate("false_positive", product_id)
                fn_rates = self._compute_consecutive_daily_rate("false_negative", product_id)

                trigger = None

                if fp_rates["consecutive_days"] >= self._retrain_consecutive_days:
                    trigger = RetrainTrigger.create(
                        trigger_reason=f"False positive rate exceeded {self._retrain_false_positive_threshold*100}% for {fp_rates['consecutive_days']} consecutive days",
                        threshold_type="false_positive_rate",
                        threshold_value=self._retrain_false_positive_threshold,
                        actual_value=fp_rates["current_rate"],
                        consecutive_days=fp_rates["consecutive_days"],
                        product_id=product_id or "all",
                        algorithm_type=algorithm_type or AlgorithmType.OBJECT_DETECTION
                    )

                elif fn_rates["consecutive_days"] >= self._retrain_consecutive_days:
                    trigger = RetrainTrigger.create(
                        trigger_reason=f"False negative rate exceeded {self._retrain_false_negative_threshold*100}% for {fn_rates['consecutive_days']} consecutive days",
                        threshold_type="false_negative_rate",
                        threshold_value=self._retrain_false_negative_threshold,
                        actual_value=fn_rates["current_rate"],
                        consecutive_days=fn_rates["consecutive_days"],
                        product_id=product_id or "all",
                        algorithm_type=algorithm_type or AlgorithmType.OBJECT_DETECTION
                    )

                if trigger:
                    self._retrain_triggers.append(trigger)
                    self._save_retrain_trigger_to_db(trigger)
                    self._execute_retrain(trigger)
                    logger.info(f"Auto retrain triggered: {trigger.trigger_id}")

                return trigger

            except Exception as e:
                logger.error(f"Failed to check retrain trigger: {e}", exc_info=True)
                return None

    def _compute_consecutive_daily_rate(self, rate_type: str,
                                        product_id: str = None) -> Dict[str, Any]:
        daily_data = self._false_positive_daily if rate_type == "false_positive" else self._false_negative_daily
        threshold = self._retrain_false_positive_threshold if rate_type == "false_positive" else self._retrain_false_negative_threshold

        today = datetime.now()
        consecutive_days = 0
        current_rate = 0.0

        for i in range(30):
            check_date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            key_prefix = rate_type
            matching_keys = [k for k in daily_data.keys() if check_date in k]

            if matching_keys:
                total_count = sum(len(daily_data[k]) for k in matching_keys)
                rate = min(total_count / 100.0, 1.0)

                if i == 0:
                    current_rate = rate

                if rate > threshold:
                    consecutive_days += 1
                else:
                    break
            else:
                break

        return {
            "consecutive_days": consecutive_days,
            "current_rate": current_rate,
            "threshold": threshold
        }

    def _execute_retrain(self, trigger: RetrainTrigger):
        trigger.status = RetrainTriggerStatus.COLLECTING
        self._save_retrain_trigger_to_db(trigger)

        collection_dir = os.path.join(
            self._collection_dir,
            f"retrain_{trigger.trigger_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        os.makedirs(collection_dir, exist_ok=True)
        trigger.collection_dir = collection_dir

        try:
            cutoff_time = time.time() - (self._retrain_sample_days * 86400)
            recent_annotations = [
                a for a in self._annotations
                if a.timestamp >= cutoff_time
            ]

            for ann in recent_annotations:
                if ann.image_path and os.path.exists(ann.image_path):
                    try:
                        import glob
                        dest_dir = os.path.join(collection_dir, ann.annotation_type.value)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, os.path.basename(ann.image_path))
                        if not os.path.exists(dest_path):
                            shutil.copy2(ann.image_path, dest_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy annotation image: {e}")

            annotation_meta_path = os.path.join(collection_dir, "annotations.json")
            with open(annotation_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    [a.to_dict() for a in recent_annotations],
                    f, ensure_ascii=False, indent=2
                )

            trigger.sample_count = len(recent_annotations)
            trigger.status = RetrainTriggerStatus.TRAINING
            self._save_retrain_trigger_to_db(trigger)

            if self._retrain_command:
                logger.info(f"Executing retrain command: {self._retrain_command}")
                import subprocess
                env = os.environ.copy()
                env["RETRAIN_COLLECTION_DIR"] = collection_dir
                env["RETRAIN_TRIGGER_ID"] = trigger.trigger_id
                env["RETRAIN_PRODUCT_ID"] = trigger.product_id
                env["RETRAIN_ALGORITHM_TYPE"] = trigger.algorithm_type.value

                result = subprocess.run(
                    self._retrain_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    env=env
                )

                trigger.training_log = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

                if result.returncode == 0:
                    trigger.status = RetrainTriggerStatus.COMPLETED
                    trigger.completed_at = time.time()
                    logger.info(f"Retrain completed for trigger {trigger.trigger_id}")
                else:
                    trigger.status = RetrainTriggerStatus.FAILED
                    trigger.completed_at = time.time()
                    trigger.training_log += f"\n\nReturn code: {result.returncode}"
                    logger.error(f"Retrain failed for trigger {trigger.trigger_id}: return code {result.returncode}")
            else:
                trigger.status = RetrainTriggerStatus.COMPLETED
                trigger.completed_at = time.time()
                trigger.training_log = "No retrain command configured, samples collected only"
                logger.info(f"Retrain samples collected for trigger {trigger.trigger_id}")

        except Exception as e:
            trigger.status = RetrainTriggerStatus.FAILED
            trigger.completed_at = time.time()
            trigger.training_log = f"Error: {str(e)}"
            logger.error(f"Retrain execution failed: {e}", exc_info=True)

        self._save_retrain_trigger_to_db(trigger)

    def list_retrain_triggers(self, status: RetrainTriggerStatus = None,
                              limit: int = 50) -> List[RetrainTrigger]:
        with self._lock:
            triggers = list(self._retrain_triggers)
            if status:
                triggers = [t for t in triggers if t.status == status]
            triggers.sort(key=lambda t: t.started_at, reverse=True)
            return triggers[:limit]

    def get_retrain_trigger(self, trigger_id: str) -> Optional[RetrainTrigger]:
        for t in self._retrain_triggers:
            if t.trigger_id == trigger_id:
                return t
        return None

    def _start_retrain_checker(self):
        if not self._retrain_auto_trigger:
            return

        def _check_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._shutdown_event.wait(self._retrain_check_interval_sec)
                    if self._shutdown_event.is_set():
                        break
                    self.check_and_trigger_retrain()
                except Exception as e:
                    logger.error(f"Retrain check loop error: {e}")

        self._retrain_check_thread = threading.Thread(target=_check_loop, daemon=True)
        self._retrain_check_thread.start()
        logger.info(f"Retrain auto-check started (interval: {self._retrain_check_interval_sec}s)")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            versions_by_status = {}
            for v in self._versions.values():
                status = v.status.value
                versions_by_status[status] = versions_by_status.get(status, 0) + 1

            annotations_by_type = {}
            for a in self._annotations:
                atype = a.annotation_type.value
                annotations_by_type[atype] = annotations_by_type.get(atype, 0) + 1

            active_ab_tests = sum(1 for t in self._ab_tests.values() if t.status == ABTestStatus.RUNNING)
            pending_retrains = sum(1 for t in self._retrain_triggers if t.status in (RetrainTriggerStatus.PENDING, RetrainTriggerStatus.COLLECTING, RetrainTriggerStatus.TRAINING))

            return {
                "enabled": self._enabled,
                "total_versions": len(self._versions),
                "versions_by_status": versions_by_status,
                "total_annotations": len(self._annotations),
                "annotations_by_type": annotations_by_type,
                "active_ab_tests": active_ab_tests,
                "total_ab_tests": len(self._ab_tests),
                "pending_retrains": pending_retrains,
                "total_retrain_triggers": len(self._retrain_triggers),
                "retrain_thresholds": {
                    "false_positive": self._retrain_false_positive_threshold,
                    "false_negative": self._retrain_false_negative_threshold,
                    "consecutive_days": self._retrain_consecutive_days
                }
            }

    def close(self):
        self._shutdown_event.set()

        if self._retrain_check_thread and self._retrain_check_thread.is_alive():
            self._retrain_check_thread.join(timeout=5)

        if self._versions_db_conn:
            try:
                self._versions_db_conn.close()
            except Exception:
                pass

        if self._annotations_db_conn:
            try:
                self._annotations_db_conn.close()
            except Exception:
                pass

        logger.info("Model manager closed")
