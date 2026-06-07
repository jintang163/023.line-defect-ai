import threading
import time
import json
import cv2
import numpy as np
from typing import Dict, Optional, List, Any
from src.messaging.base_producer import BaseMessageProducer
from src.messaging.rabbitmq_producer import RabbitMQProducer
from src.messaging.kafka_producer import KafkaProducer
from src.messaging.image_uploader import ImageUploader
from src.buffer.ring_buffer import RingBuffer
from src.buffer.local_cache import LocalCache
from src.config.settings import ConfigManager
from src.utils.schemas import ImageMessage, CapturedImage
from src.utils.logger import Logger

logger = Logger().logger


class MessageSender:
    def __init__(self, config_manager: ConfigManager,
                 ring_buffer: RingBuffer,
                 local_cache: LocalCache):
        self.config_manager = config_manager
        self.ring_buffer = ring_buffer
        self.local_cache = local_cache
        self._producer: Optional[BaseMessageProducer] = None
        self._image_uploader: Optional[ImageUploader] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        msg_cfg = config_manager.get_messaging_config()
        self.mq_type = msg_cfg.get("type", "rabbitmq")
        self.enable_compression = msg_cfg.get("compression", True)
        self.jpeg_quality = msg_cfg.get("quality", 85)
        self.embed_images_in_mq = msg_cfg.get("embed_images_in_mq", False)

        storage_cfg = msg_cfg.get("image_storage", {})
        self._image_uploader = ImageUploader(storage_cfg)

        self._init_producer()
        self.local_cache.set_retry_callback(self._retry_cached_message)

    def _init_producer(self):
        msg_cfg = self.config_manager.get_messaging_config()
        if self.mq_type == "kafka":
            self._producer = KafkaProducer(msg_cfg)
        else:
            self._producer = RabbitMQProducer(msg_cfg)

    def start(self):
        if self._image_uploader:
            logger.info(f"Image storage backend: {self._image_uploader.get_upload_stats()['backend']}")

        if self._producer and not self._producer.is_connected():
            self._producer.connect()

        if self._sender_thread is None or not self._sender_thread.is_alive():
            self._stop_event.clear()
            self._sender_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._sender_thread.start()
            logger.info("Message sender thread started")

        self.local_cache.start_retry_thread()

    def stop(self):
        self._stop_event.set()
        if self._sender_thread:
            self._sender_thread.join(timeout=10)
        self.local_cache.stop_retry_thread()
        if self._producer:
            self._producer.disconnect()
        logger.info("Message sender stopped")

    def enqueue_message(self, message: ImageMessage):
        success = self.ring_buffer.put(message, block=False)
        if not success:
            logger.warning(f"Ring buffer full, saving message {message.sequence_id} to local cache")
            self.local_cache.save(message, message.images)

    def _send_loop(self):
        while not self._stop_event.is_set():
            try:
                message = self.ring_buffer.get(block=True, timeout=1.0)
                if message is None:
                    continue

                if not self._send_message(message):
                    logger.warning(f"Send failed, saving {message.sequence_id} to local cache")
                    self.local_cache.save(message, message.images)

            except Exception as e:
                logger.error(f"Send loop error: {e}")
                time.sleep(0.1)

    def _send_message(self, message: ImageMessage) -> bool:
        if not self._producer or not self._producer.is_connected():
            if self._producer:
                self._producer.connect()
            if not self._producer or not self._producer.is_connected():
                return False

        try:
            upload_success = self._upload_and_set_urls(message)
            if not upload_success:
                logger.error(f"Failed to upload images for message {message.sequence_id}")
                return False

            success = self._producer.send(message, None)

            if success:
                logger.info(
                    f"Successfully sent message {message.sequence_id} "
                    f"with {len(message.images)} images via URL reference"
                )
            else:
                logger.warning(f"Failed to send message {message.sequence_id}")

            return success

        except Exception as e:
            logger.error(f"Error sending message {message.sequence_id}: {e}")
            return False

    def _upload_and_set_urls(self, message: ImageMessage) -> bool:
        if not self._image_uploader:
            logger.error("Image uploader not initialized")
            return False

        all_success = True
        for img in message.images:
            try:
                image_bytes = self._encode_single_image(img)
                if image_bytes is None:
                    logger.warning(f"Failed to encode image {img.image_id}, skipping")
                    all_success = False
                    continue

                metadata = {
                    "camera_id": img.camera_id,
                    "camera_position": img.camera_position,
                    "width": img.width,
                    "height": img.height,
                    "pixel_format": img.pixel_format,
                    "trigger_count": str(img.trigger_count),
                    "sequence_id": message.sequence_id
                }
                if img.metadata:
                    metadata.update({k: str(v) for k, v in img.metadata.items()})

                upload_success, url = self._image_uploader.upload_image(
                    image_data=image_bytes,
                    sequence_id=message.sequence_id,
                    camera_position=img.camera_position,
                    image_id=img.image_id,
                    metadata=metadata
                )

                if upload_success and url:
                    img.image_url = url
                    logger.debug(f"Image {img.image_id} uploaded to {url}")
                else:
                    logger.error(f"Failed to upload image {img.image_id}")
                    all_success = False

            except Exception as e:
                logger.error(f"Error uploading image {img.image_id}: {e}")
                all_success = False

        return all_success

    def _encode_single_image(self, img: CapturedImage) -> Optional[bytes]:
        try:
            data = img.processed_data if img.processed_data is not None else img.raw_data
            if data is None:
                return None

            if len(data.shape) == 2:
                data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

            if self.enable_compression:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                success, buffer = cv2.imencode(".jpg", data, encode_param)
                if success:
                    return buffer.tobytes()
            else:
                success, buffer = cv2.imencode(".png", data)
                if success:
                    return buffer.tobytes()

        except Exception as e:
            logger.error(f"Error encoding image {img.image_id}: {e}")

        return None

    def _retry_cached_message(self, cached_metadata: Dict[str, Any]) -> bool:
        if not self._producer or not self._producer.is_connected():
            return False

        try:
            sequence_id = cached_metadata["sequence_id"]
            image_paths = cached_metadata.get("image_paths", {})
            images_info = cached_metadata.get("images", [])

            image_urls: Dict[str, str] = {}
            for img_id, img_path in image_paths.items():
                try:
                    img_data = cv2.imread(img_path)
                    if img_data is None:
                        logger.warning(f"Could not read cached image {img_path}")
                        continue

                    img_info = next((i for i in images_info if i.get("image_id") == img_id), {})
                    camera_position = img_info.get("camera_position", "unknown")

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    success, buffer = cv2.imencode(".jpg", img_data, encode_param)
                    if not success:
                        continue

                    metadata = {
                        "camera_id": img_info.get("camera_id", ""),
                        "camera_position": camera_position,
                        "width": str(img_info.get("width", 0)),
                        "height": str(img_info.get("height", 0)),
                        "retry": "true"
                    }

                    upload_success, url = self._image_uploader.upload_image(
                        image_data=buffer.tobytes(),
                        sequence_id=sequence_id,
                        camera_position=camera_position,
                        image_id=img_id,
                        metadata=metadata
                    )

                    if upload_success and url:
                        image_urls[img_id] = url
                        for img in images_info:
                            if img.get("image_id") == img_id:
                                img["image_url"] = url

                except Exception as e:
                    logger.error(f"Error processing cached image {img_path}: {e}")

            if not image_urls:
                logger.warning(f"No images uploaded for cached message {sequence_id}")
                return False

            for img in images_info:
                if img.get("image_id") not in image_urls:
                    img["image_url"] = image_urls.get(img.get("image_id", ""))

            message_dict = {
                "sequence_id": sequence_id,
                "timestamp": cached_metadata.get("timestamp", time.time()),
                "product_id": cached_metadata.get("product_id"),
                "line_id": cached_metadata.get("line_id", "line-001"),
                "images": images_info
            }

            body = json.dumps(message_dict).encode("utf-8")
            success = self._producer.send_raw(body, sequence_id)

            if success:
                logger.info(f"Successfully retried cached message {sequence_id} with {len(image_urls)} image URLs")

            return success

        except Exception as e:
            logger.error(f"Error retrying cached message: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "mq_type": self.mq_type,
            "mq_connected": self._producer.is_connected() if self._producer else False,
            "mq_send_count": self._producer.send_count if self._producer else 0,
            "mq_fail_count": self._producer.fail_count if self._producer else 0,
            "ring_buffer_size": len(self.ring_buffer),
            "ring_buffer_overflow": self.ring_buffer.overflow_count,
            "embed_images_in_mq": self.embed_images_in_mq,
            **self.local_cache.get_cache_stats()
        }

        if self._image_uploader:
            stats["image_storage"] = self._image_uploader.get_upload_stats()

        return stats

    def reconnect(self) -> bool:
        if self._producer:
            self._producer.disconnect()
            return self._producer.connect()
        return False

    def reconnect_storage(self) -> bool:
        if self._image_uploader:
            return self._image_uploader.reconnect()
        return False
