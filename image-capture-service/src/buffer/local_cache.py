import os
import json
import time
import threading
import pickle
from typing import Optional, List, Dict, Any
from pathlib import Path
from src.utils.schemas import ImageMessage, CapturedImage
from src.utils.logger import Logger
import numpy as np
import cv2

logger = Logger().logger


class LocalCache:
    def __init__(self, cache_dir: str = "./data/cache", max_size_gb: int = 10,
                 retry_interval: int = 30, max_retry: int = 10):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.retry_interval = retry_interval
        self.max_retry = max_retry
        self._lock = threading.Lock()
        self._retry_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pending_files: List[Path] = []
        self._callback = None

        self._init_directories()
        self._scan_pending_files()

    def _init_directories(self):
        (self.cache_dir / "pending").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "temp").mkdir(parents=True, exist_ok=True)
        logger.info(f"Local cache initialized at {self.cache_dir}")

    def _scan_pending_files(self):
        pending_dir = self.cache_dir / "pending"
        self._pending_files = sorted(
            list(pending_dir.glob("*.json")),
            key=lambda x: x.stat().st_mtime
        )
        logger.info(f"Found {len(self._pending_files)} pending cache files")

    def set_retry_callback(self, callback):
        self._callback = callback

    def save(self, message: ImageMessage, images: List[CapturedImage]) -> bool:
        self._check_disk_space()

        timestamp = int(time.time() * 1000)
        base_filename = f"{message.sequence_id}_{timestamp}"

        try:
            with self._lock:
                image_paths = {}
                for img in images:
                    if img.processed_data is not None:
                        img_path = self._save_image(img, base_filename)
                        image_paths[img.image_id] = img_path

                metadata = {
                    "sequence_id": message.sequence_id,
                    "timestamp": message.timestamp,
                    "product_id": message.product_id,
                    "line_id": message.line_id,
                    "image_paths": image_paths,
                    "retry_count": 0,
                    "created_at": time.time(),
                    "images": [
                        {
                            "image_id": img.image_id,
                            "camera_id": img.camera_id,
                            "camera_position": img.camera_position,
                            "timestamp": img.timestamp,
                            "width": img.width,
                            "height": img.height,
                            "pixel_format": img.pixel_format,
                            "trigger_count": img.trigger_count,
                            "metadata": img.metadata
                        }
                        for img in images
                    ]
                }

                pending_file = self.cache_dir / "pending" / f"{base_filename}.json"
                temp_file = self.cache_dir / "temp" / f"{base_filename}.json.tmp"

                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                os.replace(temp_file, pending_file)
                self._pending_files.append(pending_file)

                logger.info(f"Saved message {message.sequence_id} to local cache")
                return True

        except Exception as e:
            logger.error(f"Failed to save to local cache: {e}")
            return False

    def _save_image(self, img: CapturedImage, base_filename: str) -> str:
        images_dir = self.cache_dir / "images"
        filename = f"{base_filename}_{img.camera_position}.jpg"
        filepath = images_dir / filename

        data = img.processed_data if img.processed_data is not None else img.raw_data
        if data is not None:
            if len(data.shape) == 2:
                data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(filepath), data, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return str(filepath)

    def _check_disk_space(self):
        try:
            total_size = self._get_cache_size()
            if total_size > self.max_size_bytes:
                logger.warning(f"Cache size {total_size / 1e9:.2f}GB exceeds limit {self.max_size_bytes / 1e9:.2f}GB")
                self._purge_old_files(total_size - self.max_size_bytes)
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")

    def _get_cache_size(self) -> int:
        total = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def _purge_old_files(self, bytes_to_free: int):
        files = []
        pending_dir = self.cache_dir / "pending"
        images_dir = self.cache_dir / "images"

        for directory in [pending_dir, images_dir]:
            for f in directory.glob("*"):
                if f.is_file():
                    files.append((f, f.stat().st_mtime))

        files.sort(key=lambda x: x[1])

        freed = 0
        for f, _ in files:
            try:
                size = f.stat().st_size
                f.unlink()
                freed += size
                if freed >= bytes_to_free:
                    break
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")

        logger.info(f"Purged {freed / 1e6:.2f}MB from cache")
        self._scan_pending_files()

    def load_pending(self, max_count: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            items = []
            files_to_process = self._pending_files[:max_count]

            for pending_file in files_to_process:
                try:
                    with open(pending_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    items.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to load pending file {pending_file}: {e}")
                    try:
                        pending_file.unlink()
                    except:
                        pass

            return items

    def mark_sent(self, sequence_id: str):
        with self._lock:
            for i, pending_file in enumerate(self._pending_files):
                if sequence_id in pending_file.name:
                    try:
                        self._delete_cached_files(sequence_id)
                        pending_file.unlink()
                        self._pending_files.pop(i)
                        logger.info(f"Marked {sequence_id} as sent, removed from cache")
                    except Exception as e:
                        logger.error(f"Failed to remove cache files for {sequence_id}: {e}")
                    break

    def mark_failed(self, sequence_id: str):
        with self._lock:
            for pending_file in self._pending_files:
                if sequence_id in pending_file.name:
                    try:
                        with open(pending_file, "r+", encoding="utf-8") as f:
                            metadata = json.load(f)
                            metadata["retry_count"] = metadata.get("retry_count", 0) + 1
                            metadata["last_retry_at"] = time.time()

                            if metadata["retry_count"] >= self.max_retry:
                                logger.error(f"Max retries reached for {sequence_id}, deleting")
                                self._delete_cached_files(sequence_id)
                                pending_file.unlink()
                                self._pending_files.remove(pending_file)
                            else:
                                f.seek(0)
                                json.dump(metadata, f, ensure_ascii=False, indent=2)
                                f.truncate()
                    except Exception as e:
                        logger.error(f"Failed to update retry count for {sequence_id}: {e}")
                    break

    def _delete_cached_files(self, sequence_id: str):
        images_dir = self.cache_dir / "images"
        for f in images_dir.glob(f"{sequence_id}_*"):
            try:
                f.unlink()
            except Exception as e:
                logger.error(f"Failed to delete image {f}: {e}")

    def start_retry_thread(self):
        if self._retry_thread is None or not self._retry_thread.is_alive():
            self._stop_event.clear()
            self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
            self._retry_thread.start()
            logger.info("Local cache retry thread started")

    def stop_retry_thread(self):
        self._stop_event.set()
        if self._retry_thread:
            self._retry_thread.join(timeout=5)
        logger.info("Local cache retry thread stopped")

    def _retry_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._callback and self._pending_files:
                    pending_items = self.load_pending(max_count=5)
                    for item in pending_items:
                        try:
                            last_retry = item.get("last_retry_at", 0)
                            if time.time() - last_retry < self.retry_interval:
                                continue

                            if self._callback(item):
                                self.mark_sent(item["sequence_id"])
                            else:
                                self.mark_failed(item["sequence_id"])
                        except Exception as e:
                            logger.error(f"Retry failed for {item.get('sequence_id')}: {e}")
                            self.mark_failed(item.get("sequence_id", ""))

            except Exception as e:
                logger.error(f"Retry loop error: {e}")

            self._stop_event.wait(self.retry_interval)

    def get_pending_count(self) -> int:
        return len(self._pending_files)

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "pending_count": self.get_pending_count(),
            "total_size_bytes": self._get_cache_size(),
            "max_size_bytes": self.max_size_bytes,
            "cache_dir": str(self.cache_dir)
        }
