import threading
import time
import json
import cv2
import numpy as np
from typing import Dict, Optional, List, Any
from src.messaging.base_producer import BaseMessageProducer
from src.messaging.rabbitmq_producer import RabbitMQProducer
from src.messaging.kafka_producer import KafkaProducer
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
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        msg_cfg = config_manager.get_messaging_config()
        self.mq_type = msg_cfg.get("type", "rabbitmq")
        self.enable_compression = msg_cfg.get("compression", True)
        self.jpeg_quality = msg_cfg.get("quality", 85)

        self._init_producer()
        self.local_cache.set_retry_callback(self._retry_cached_message)

    def _init_producer(self):
        msg_cfg = self.config_manager.get_messaging_config()
        if self.mq_type == "kafka":
            self._producer = KafkaProducer(msg_cfg)
        else:
            self._producer = RabbitMQProducer(msg_cfg)

    def start(self):
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
            images_data = self._encode_images(message.images)
            success = self._producer.send(message, images_data)

            if success:
                logger.info(f"Successfully sent message {message.sequence_id}")
            else:
                logger.warning(f"Failed to send message {message.sequence_id}")

            return success

        except Exception as e:
            logger.error(f"Error sending message {message.sequence_id}: {e}")
            return False

    def _encode_images(self, images: List[CapturedImage]) -> Dict[str, bytes]:
        encoded = {}
        for img in images:
            try:
                data = img.processed_data if img.processed_data is not None else img.raw_data
                if data is None:
                    continue

                if len(data.shape) == 2:
                    data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

                if self.enable_compression:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    success, buffer = cv2.imencode(".jpg", data, encode_param)
                    if success:
                        encoded[img.image_id] = buffer.tobytes()
                else:
                    encoded[img.image_id] = data.tobytes()

            except Exception as e:
                logger.error(f"Error encoding image {img.image_id}: {e}")

        return encoded

    def _retry_cached_message(self, cached_metadata: Dict[str, Any]) -> bool:
        if not self._producer or not self._producer.is_connected():
            return False

        try:
            sequence_id = cached_metadata["sequence_id"]
            image_paths = cached_metadata.get("image_paths", {})

            images_data = {}
            for img_id, img_path in image_paths.items():
                try:
                    img_data = cv2.imread(img_path)
                    if img_data is not None:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                        success, buffer = cv2.imencode(".jpg", img_data, encode_param)
                        if success:
                            images_data[img_id] = buffer.tobytes()
                except Exception as e:
                    logger.error(f"Error loading cached image {img_path}: {e}")

            if not images_data:
                logger.warning(f"No images loaded for cached message {sequence_id}")
                return False

            message_dict = {
                "sequence_id": sequence_id,
                "timestamp": cached_metadata.get("timestamp", time.time()),
                "product_id": cached_metadata.get("product_id"),
                "line_id": cached_metadata.get("line_id", "line-001"),
                "images": cached_metadata.get("images", []),
                "image_data": {
                    img_id: list(data) for img_id, data in images_data.items()
                }
            }

            body = json.dumps(message_dict).encode("utf-8")
            success = self._producer.send_raw(body, sequence_id)

            if success:
                logger.info(f"Successfully retried cached message {sequence_id}")

            return success

        except Exception as e:
            logger.error(f"Error retrying cached message: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mq_type": self.mq_type,
            "connected": self._producer.is_connected() if self._producer else False,
            "send_count": self._producer.send_count if self._producer else 0,
            "fail_count": self._producer.fail_count if self._producer else 0,
            "ring_buffer_size": len(self.ring_buffer),
            "ring_buffer_overflow": self.ring_buffer.overflow_count,
            **self.local_cache.get_cache_stats()
        }

    def reconnect(self) -> bool:
        if self._producer:
            self._producer.disconnect()
            return self._producer.connect()
        return False
