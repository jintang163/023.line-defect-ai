from typing import Dict, Any, Optional, List
import threading
import os
import time
from datetime import datetime

from src.utils.logger import Logger

logger = Logger("image_archive", "INFO", "./logs/defect-detection.log").logger

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("minio package not available, MinIO storage disabled")

import cv2
import numpy as np


class ImageArchive:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)
        self._storage_type = config.get("storage_type", "local")
        self._local_dir = config.get("local_dir", "./data/images")
        self._minio_config = config.get("minio", {})
        self._retention_days = config.get("retention_days", 90)
        self._auto_cleanup = config.get("auto_cleanup", False)
        self._cleanup_interval_hours = config.get("cleanup_interval_hours", 24)
        self._thumbnail_max_size = config.get("thumbnail_max_size", 200)
        self._compression_quality = config.get("compression_quality", 85)
        self._lock = threading.RLock()
        self._minio_client: Optional[Any] = None
        self._cleanup_timer: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self._enabled:
            if self._storage_type == "local":
                os.makedirs(self._local_dir, exist_ok=True)
                logger.info(f"Image archive initialized with local storage: {self._local_dir}")
            elif self._storage_type == "minio":
                self._init_minio()

            if self._auto_cleanup:
                self._start_cleanup_timer()

    def _init_minio(self):
        if not MINIO_AVAILABLE:
            logger.error("MinIO package not available, cannot use MinIO storage")
            self._enabled = False
            return

        try:
            self._minio_client = Minio(
                endpoint=self._minio_config.get("endpoint", "localhost:9000"),
                access_key=self._minio_config.get("access_key", ""),
                secret_key=self._minio_config.get("secret_key", ""),
                secure=self._minio_config.get("use_ssl", False)
            )
            bucket = self._minio_config.get("bucket", "defect-images")
            if not self._minio_client.bucket_exists(bucket):
                self._minio_client.make_bucket(bucket)
            logger.info(f"MinIO storage initialized: {self._minio_config.get('endpoint')}/{bucket}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}", exc_info=True)
            self._enabled = False

    def archive_image(self, detection_id: str, original_image: bytes, annotated_image: bytes,
                      product_id: str, timestamp: float) -> Dict[str, str]:
        if not self._enabled:
            return {}

        with self._lock:
            try:
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

                original_name = f"{detection_id}_original.jpg"
                annotated_name = f"{detection_id}_annotated.jpg"
                thumbnail_name = f"{detection_id}_thumb.jpg"

                original_np = np.frombuffer(original_image, dtype=np.uint8)
                original_decoded = cv2.imdecode(original_np, cv2.IMREAD_COLOR)
                _, original_encoded = cv2.imencode(".jpg", original_decoded,
                                                   [cv2.IMWRITE_JPEG_QUALITY, self._compression_quality])
                original_bytes = original_encoded.tobytes()

                annotated_np = np.frombuffer(annotated_image, dtype=np.uint8)
                annotated_decoded = cv2.imdecode(annotated_np, cv2.IMREAD_COLOR)
                _, annotated_encoded = cv2.imencode(".jpg", annotated_decoded,
                                                    [cv2.IMWRITE_JPEG_QUALITY, self._compression_quality])
                annotated_bytes = annotated_encoded.tobytes()

                thumbnail_decoded = self._create_thumbnail(annotated_decoded)
                _, thumbnail_encoded = cv2.imencode(".jpg", thumbnail_decoded,
                                                    [cv2.IMWRITE_JPEG_QUALITY, self._compression_quality])
                thumbnail_bytes = thumbnail_encoded.tobytes()

                if self._storage_type == "local":
                    dir_path = os.path.join(self._local_dir, product_id, date_str)
                    os.makedirs(dir_path, exist_ok=True)

                    original_path = os.path.join(dir_path, original_name)
                    annotated_path = os.path.join(dir_path, annotated_name)
                    thumbnail_path = os.path.join(dir_path, thumbnail_name)

                    with open(original_path, "wb") as f:
                        f.write(original_bytes)
                    with open(annotated_path, "wb") as f:
                        f.write(annotated_bytes)
                    with open(thumbnail_path, "wb") as f:
                        f.write(thumbnail_bytes)

                    logger.info(f"Images archived locally: {dir_path}")

                    return {
                        "original_path": original_path,
                        "annotated_path": annotated_path,
                        "thumbnail_path": thumbnail_path
                    }
                elif self._storage_type == "minio":
                    bucket = self._minio_config.get("bucket", "defect-images")
                    prefix = f"{product_id}/{date_str}"

                    original_path = f"{prefix}/{original_name}"
                    annotated_path = f"{prefix}/{annotated_name}"
                    thumbnail_path = f"{prefix}/{thumbnail_name}"

                    self._minio_client.put_object(bucket, original_path,
                                                  __import__("io").BytesIO(original_bytes), len(original_bytes),
                                                  content_type="image/jpeg")
                    self._minio_client.put_object(bucket, annotated_path,
                                                  __import__("io").BytesIO(annotated_bytes), len(annotated_bytes),
                                                  content_type="image/jpeg")
                    self._minio_client.put_object(bucket, thumbnail_path,
                                                  __import__("io").BytesIO(thumbnail_bytes), len(thumbnail_bytes),
                                                  content_type="image/jpeg")

                    logger.info(f"Images archived to MinIO: {prefix}")

                    return {
                        "original_path": original_path,
                        "annotated_path": annotated_path,
                        "thumbnail_path": thumbnail_path
                    }

                return {}
            except Exception as e:
                logger.error(f"Failed to archive image: {e}", exc_info=True)
                return {}

    def _create_thumbnail(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > self._thumbnail_max_size:
            scale = self._thumbnail_max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def get_image(self, image_path: str) -> Optional[bytes]:
        if not self._enabled:
            return None

        with self._lock:
            try:
                if self._storage_type == "local":
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            return f.read()
                    return None
                elif self._storage_type == "minio":
                    bucket = self._minio_config.get("bucket", "defect-images")
                    response = self._minio_client.get_object(bucket, image_path)
                    data = response.read()
                    response.close()
                    response.release_conn()
                    return data
                return None
            except Exception as e:
                logger.error(f"Failed to get image {image_path}: {e}", exc_info=True)
                return None

    def get_thumbnail(self, detection_id: str, product_id: str) -> Optional[bytes]:
        if not self._enabled:
            return None

        with self._lock:
            try:
                if self._storage_type == "local":
                    product_dir = os.path.join(self._local_dir, product_id)
                    if not os.path.exists(product_dir):
                        return None

                    for date_dir_name in sorted(os.listdir(product_dir), reverse=True):
                        date_dir = os.path.join(product_dir, date_dir_name)
                        if not os.path.isdir(date_dir):
                            continue
                        thumb_path = os.path.join(date_dir, f"{detection_id}_thumb.jpg")
                        if os.path.exists(thumb_path):
                            with open(thumb_path, "rb") as f:
                                return f.read()
                    return None
                elif self._storage_type == "minio":
                    bucket = self._minio_config.get("bucket", "defect-images")
                    prefix = f"{product_id}/"
                    objects = list(self._minio_client.list_objects(bucket, prefix=prefix, recursive=True))
                    for obj in objects:
                        if obj.object_name.endswith(f"{detection_id}_thumb.jpg"):
                            response = self._minio_client.get_object(bucket, obj.object_name)
                            data = response.read()
                            response.close()
                            response.release_conn()
                            return data
                    return None
                return None
            except Exception as e:
                logger.error(f"Failed to get thumbnail for {detection_id}: {e}", exc_info=True)
                return None

    def delete_images(self, image_paths: List[str]) -> int:
        if not self._enabled:
            return 0

        with self._lock:
            deleted = 0
            for path in image_paths:
                try:
                    if self._storage_type == "local":
                        if os.path.exists(path):
                            os.remove(path)
                            deleted += 1
                    elif self._storage_type == "minio":
                        bucket = self._minio_config.get("bucket", "defect-images")
                        self._minio_client.remove_object(bucket, path)
                        deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete image {path}: {e}", exc_info=True)
            return deleted

    def cleanup_expired(self, retention_days: Optional[int] = None) -> int:
        if not self._enabled or self._storage_type != "local":
            return 0

        days = retention_days if retention_days is not None else self._retention_days

        with self._lock:
            try:
                cutoff = datetime.now()
                deleted = 0

                if not os.path.exists(self._local_dir):
                    return 0

                for product_dir_name in os.listdir(self._local_dir):
                    product_dir = os.path.join(self._local_dir, product_dir_name)
                    if not os.path.isdir(product_dir):
                        continue

                    for date_dir_name in os.listdir(product_dir):
                        date_dir = os.path.join(product_dir, date_dir_name)
                        if not os.path.isdir(date_dir):
                            continue

                        try:
                            dir_date = datetime.strptime(date_dir_name, "%Y-%m-%d")
                            age_days = (cutoff - dir_date).days
                            if age_days > days:
                                import shutil
                                shutil.rmtree(date_dir)
                                deleted += 1
                                logger.info(f"Cleaned up expired directory: {date_dir}")
                        except ValueError:
                            continue

                return deleted
            except Exception as e:
                logger.error(f"Failed to cleanup expired images: {e}", exc_info=True)
                return 0

    def _start_cleanup_timer(self):
        if self._cleanup_timer is not None and self._cleanup_timer.is_alive():
            return

        def _cleanup_loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(self._cleanup_interval_hours * 3600)
                if not self._stop_event.is_set():
                    try:
                        count = self.cleanup_expired()
                        if count > 0:
                            logger.info(f"Auto-cleanup removed {count} expired directories")
                    except Exception as e:
                        logger.error(f"Auto-cleanup error: {e}", exc_info=True)

        self._stop_event.clear()
        self._cleanup_timer = threading.Thread(target=_cleanup_loop, daemon=True)
        self._cleanup_timer.start()
        logger.info(f"Auto-cleanup timer started, interval: {self._cleanup_interval_hours} hours")

    def close(self):
        self._stop_event.set()
        if self._cleanup_timer is not None and self._cleanup_timer.is_alive():
            self._cleanup_timer.join(timeout=5)
        self._cleanup_timer = None
        logger.info("Image archive closed")
