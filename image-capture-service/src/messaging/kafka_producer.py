import json
import time
from typing import Dict, Any, Optional
from src.messaging.base_producer import BaseMessageProducer
from src.utils.schemas import ImageMessage
from src.utils.logger import Logger

logger = Logger().logger


class KafkaProducer(BaseMessageProducer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        kafka_cfg = config.get("kafka", {})
        self.brokers = kafka_cfg.get("brokers", ["localhost:9092"])
        self.topic = kafka_cfg.get("topic", "defect-images")
        self.client_id = kafka_cfg.get("client_id", "capture-service")
        self.compression_type = kafka_cfg.get("compression_type", "gzip")
        self.acks = kafka_cfg.get("acks", 1)

        self._producer = None
        self._last_reconnect = 0
        self._reconnect_interval = 5

    def _try_import_kafka(self) -> bool:
        try:
            from kafka import KafkaProducer as KProducer
            self._kafka_producer_class = KProducer
            return True
        except ImportError:
            logger.warning("kafka-python not installed, Kafka producer will use mock mode")
            return False

    def connect(self) -> bool:
        if not self._try_import_kafka():
            self._is_connected = True
            logger.info("Kafka producer connected in mock mode")
            return True

        try:
            self._producer = self._kafka_producer_class(
                bootstrap_servers=self.brokers,
                client_id=self.client_id,
                compression_type=self.compression_type,
                acks=self.acks,
                retries=3,
                retry_backoff_ms=1000,
                value_serializer=lambda v: v,
                key_serializer=lambda k: k.encode("utf-8") if k else None
            )

            self._is_connected = True
            logger.info(f"Connected to Kafka at {self.brokers}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._is_connected = False
            return False

    def _reconnect_if_needed(self) -> bool:
        if self._is_connected:
            return True

        now = time.time()
        if now - self._last_reconnect < self._reconnect_interval:
            return False

        logger.info("Attempting to reconnect to Kafka...")
        self._last_reconnect = now
        return self.connect()

    def disconnect(self):
        try:
            if self._producer:
                self._producer.flush(timeout=10)
                self._producer.close()
        except Exception as e:
            logger.error(f"Error disconnecting Kafka: {e}")
        finally:
            self._producer = None
            self._is_connected = False
            logger.info("Disconnected from Kafka")

    def send(self, message: ImageMessage, images_data: Optional[Dict[str, bytes]] = None) -> bool:
        if not self._reconnect_if_needed():
            self._increment_fail_count()
            return False

        try:
            message_dict = message.to_dict()
            if images_data:
                message_dict["image_data"] = {
                    img_id: list(data) for img_id, data in images_data.items()
                }

            body = json.dumps(message_dict).encode("utf-8")

            if self._producer:
                future = self._producer.send(
                    self.topic,
                    key=message.sequence_id,
                    value=body
                )
                future.get(timeout=10)
            else:
                logger.debug(f"Mock Kafka send: {message.sequence_id} ({len(body)} bytes)")

            self._increment_send_count()
            logger.debug(f"Sent message {message.sequence_id} to Kafka")
            return True

        except Exception as e:
            logger.error(f"Kafka send error: {e}")
            self._is_connected = False
            self._increment_fail_count()
            return False

    def send_raw(self, body: bytes, message_id: str) -> bool:
        if not self._reconnect_if_needed():
            self._increment_fail_count()
            return False

        try:
            if self._producer:
                future = self._producer.send(
                    self.topic,
                    key=message_id,
                    value=body
                )
                future.get(timeout=10)
            else:
                logger.debug(f"Mock Kafka raw send: {message_id} ({len(body)} bytes)")

            self._increment_send_count()
            return True

        except Exception as e:
            logger.error(f"Kafka raw send error: {e}")
            self._is_connected = False
            self._increment_fail_count()
            return False

    def flush(self):
        if self._producer:
            self._producer.flush()
