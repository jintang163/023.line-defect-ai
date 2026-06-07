import os
import io
import time
import threading
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from src.utils.logger import Logger

logger = Logger().logger


class ImageUploader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._client = None
        self._is_connected = False
        self._lock = threading.Lock()
        self._endpoint = config.get("endpoint", "localhost:9000")
        self._access_key = config.get("access_key", "minioadmin")
        self._secret_key = config.get("secret_key", "minioadmin")
        self._bucket = config.get("bucket", "defect-images")
        self._secure = config.get("secure", False)
        self._use_local_fallback = config.get("use_local_fallback", True)
        self._local_storage_path = config.get("local_storage_path", "./data/images")
        self._public_base_url = config.get("public_base_url", "http://localhost:9000")

        os.makedirs(self._local_storage_path, exist_ok=True)
        self._init_client()

    def _init_client(self):
        if not self._try_import_minio():
            if self._use_local_fallback:
                logger.warning("MinIO SDK not available, using local storage fallback")
            return
        self._connect()

    def _try_import_minio(self) -> bool:
        try:
            from minio import Minio
            self._Minio = Minio
            return True
        except ImportError:
            logger.warning("minio SDK not installed")
            return False

    def _connect(self) -> bool:
        try:
            self._client = self._Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure
            )

            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                policy = self._get_public_policy(self._bucket)
                self._client.set_bucket_policy(self._bucket, policy)

            self._is_connected = True
            logger.info(f"Connected to MinIO at {self._endpoint}, bucket: {self._bucket}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            self._is_connected = False
            return False

    def _get_public_policy(self, bucket: str) -> str:
        import json
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket}/*"]
                }
            ]
        }
        return json.dumps(policy)

    def _generate_object_name(self, sequence_id: str, camera_position: str, image_id: str) -> str:
        now = datetime.now()
        date_path = now.strftime("%Y/%m/%d")
        timestamp = int(now.timestamp() * 1000)
        ext = "jpg"
        return f"{date_path}/{sequence_id}_{camera_position}_{timestamp}_{image_id[:8]}.{ext}"

    def upload_image(self, image_data: bytes, sequence_id: str,
                     camera_position: str, image_id: str,
                     metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        object_name = self._generate_object_name(sequence_id, camera_position, image_id)

        if self._is_connected and self._client:
            return self._upload_to_minio(image_data, object_name, metadata)
        elif self._use_local_fallback:
            return self._upload_to_local(image_data, object_name, metadata)
        else:
            logger.error("No upload backend available and local fallback disabled")
            return False, None

    def _upload_to_minio(self, image_data: bytes, object_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        try:
            with self._lock:
                if not self._is_connected:
                    self._connect()
                    if not self._is_connected:
                        if self._use_local_fallback:
                            return self._upload_to_local(image_data, object_name, metadata)
                        return False, None

                file_data = io.BytesIO(image_data)
                file_size = len(image_data)

                extra_args = {"Content-Type": "image/jpeg"}
                if metadata:
                    for k, v in metadata.items():
                        extra_args[f"X-Amz-Meta-{k}"] = str(v)

                self._client.put_object(
                    self._bucket,
                    object_name,
                    file_data,
                    file_size,
                    **extra_args
                )

                url = f"{self._public_base_url}/{self._bucket}/{object_name}"
                logger.debug(f"Uploaded image to MinIO: {url}")
                return True, url

        except Exception as e:
            logger.error(f"MinIO upload failed: {e}")
            if self._use_local_fallback:
                return self._upload_to_local(image_data, object_name, metadata)
            return False, None

    def _upload_to_local(self, image_data: bytes, object_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        try:
            local_path = os.path.join(self._local_storage_path, object_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(image_data)

            if metadata:
                meta_path = local_path + ".meta"
                with open(meta_path, "w", encoding="utf-8") as f:
                    import json
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            url = f"file://{os.path.abspath(local_path)}"
            logger.debug(f"Saved image to local storage: {url}")
            return True, url

        except Exception as e:
            logger.error(f"Local storage upload failed: {e}")
            return False, None

    def is_connected(self) -> bool:
        return self._is_connected

    def reconnect(self) -> bool:
        if hasattr(self, '_Minio'):
            return self._connect()
        return False

    def get_upload_stats(self) -> Dict[str, Any]:
        return {
            "backend": "minio" if self._is_connected else "local",
            "connected": self._is_connected,
            "endpoint": self._endpoint,
            "bucket": self._bucket,
            "local_fallback_enabled": self._use_local_fallback,
            "local_storage_path": self._local_storage_path
        }
