from typing import Optional, Callable, Dict, Any
import threading
import time
import json
import numpy as np
import cv2
import base64

from src.utils.schemas import ImageData
from src.utils.logger import Logger

logger = Logger().logger

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    logger.warning("pika not available, RabbitMQ consumer disabled")

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not available, Kafka consumer disabled")


class MessageConsumer:
    def __init__(self, config: Dict[str, Any],
                 callback: Optional[Callable[[ImageData], None]] = None):
        self._config = config
        self._callback = callback
        self._consumer = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._type = config.get("type", "rabbitmq")
        self._host = config.get("host", "localhost")
        self._port = config.get("port", 5672)
        self._username = config.get("username", "guest")
        self._password = config.get("password", "guest")
        self._queue = config.get("queue", "image-capture-queue")
        self._topic = config.get("topic", "image-capture-topic")
        self._exchange = config.get("exchange", "line-defect-exchange")
        self._routing_key = config.get("routing_key", "image.captured")

        self._stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_errors": 0
        }

    def set_callback(self, callback: Callable[[ImageData], None]):
        self._callback = callback

    def connect(self) -> bool:
        try:
            if self._type == "rabbitmq":
                return self._connect_rabbitmq()
            elif self._type == "kafka":
                return self._connect_kafka()
            else:
                logger.error(f"Unsupported message queue type: {self._type}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to message queue: {e}", exc_info=True)
            return False

    def _connect_rabbitmq(self) -> bool:
        if not RABBITMQ_AVAILABLE:
            logger.error("RabbitMQ not available, please install pika")
            return False

        try:
            credentials = pika.PlainCredentials(self._username, self._password)
            parameters = pika.ConnectionParameters(
                host=self._host,
                port=self._port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )

            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()

            self._channel.exchange_declare(
                exchange=self._exchange,
                exchange_type='direct',
                durable=True
            )

            self._channel.queue_declare(
                queue=self._queue,
                durable=True,
                arguments={'x-message-ttl': 60000}
            )

            self._channel.queue_bind(
                exchange=self._exchange,
                queue=self._queue,
                routing_key=self._routing_key
            )

            self._channel.basic_qos(prefetch_count=1)

            logger.info(f"Connected to RabbitMQ at {self._host}:{self._port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}", exc_info=True)
            return False

    def _connect_kafka(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available, please install kafka-python")
            return False

        try:
            self._consumer = KafkaConsumer(
                self._topic,
                bootstrap_servers=f"{self._host}:{self._port}",
                group_id='defect-detection-group',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )

            logger.info(f"Connected to Kafka at {self._host}:{self._port}, topic: {self._topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}", exc_info=True)
            return False

    def start(self):
        if self._is_running:
            logger.warning("Consumer already running")
            return

        if not self.connect():
            logger.error("Failed to start consumer: connection failed")
            return

        self._is_running = True
        self._stop_event.clear()

        if self._type == "rabbitmq":
            self._thread = threading.Thread(target=self._run_rabbitmq, daemon=True)
        elif self._type == "kafka":
            self._thread = threading.Thread(target=self._run_kafka, daemon=True)

        self._thread.start()
        logger.info("Message consumer started")

    def _run_rabbitmq(self):
        try:
            self._channel.basic_consume(
                queue=self._queue,
                on_message_callback=self._rabbitmq_callback,
                auto_ack=False
            )

            while not self._stop_event.is_set() and self._connection.is_open:
                self._connection.process_data_events(time_limit=1)

        except Exception as e:
            logger.error(f"RabbitMQ consumer error: {e}", exc_info=True)
        finally:
            self._is_running = False
            logger.info("RabbitMQ consumer stopped")

    def _rabbitmq_callback(self, ch, method, properties, body):
        try:
            message = json.loads(body.decode('utf-8'))
            self._stats["total_received"] += 1

            image_data = self._parse_message(message)
            if image_data and self._callback:
                self._callback(image_data)
                self._stats["total_processed"] += 1

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"Error processing RabbitMQ message: {e}", exc_info=True)
            self._stats["total_errors"] += 1
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _run_kafka(self):
        try:
            for message in self._consumer:
                if self._stop_event.is_set():
                    break

                try:
                    self._stats["total_received"] += 1

                    image_data = self._parse_message(message.value)
                    if image_data and self._callback:
                        self._callback(image_data)
                        self._stats["total_processed"] += 1

                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}", exc_info=True)
                    self._stats["total_errors"] += 1

        except Exception as e:
            logger.error(f"Kafka consumer error: {e}", exc_info=True)
        finally:
            self._is_running = False
            logger.info("Kafka consumer stopped")

    def _parse_message(self, message: Dict[str, Any]) -> Optional[ImageData]:
        try:
            images_info = message.get("images", [])
            image_data_map = message.get("image_data", {})

            if not images_info:
                logger.warning("No images in message")
                return None

            first_image = images_info[0]
            image_id = first_image.get("image_id", "")
            camera_id = first_image.get("camera_id", "")
            camera_position = first_image.get("camera_position", "")
            sequence_id = message.get("sequence_id", "")
            timestamp = first_image.get("timestamp", time.time())

            image_bytes = None
            if image_id in image_data_map:
                image_bytes = image_data_map[image_id]
                if isinstance(image_bytes, str):
                    image_bytes = base64.b64decode(image_bytes)

            if image_bytes is None:
                image_url = first_image.get("image_url", "")
                if image_url:
                    image_bytes = self._fetch_image(image_url)

            if image_bytes is None:
                logger.warning(f"No image data for image_id: {image_id}")
                return None

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error(f"Failed to decode image: {image_id}")
                return None

            metadata = {
                "line_id": message.get("line_id", ""),
                "product_id": message.get("product_id", ""),
                "original_timestamp": timestamp,
                "camera_metadata": first_image.get("metadata", {})
            }

            return ImageData.create(
                camera_id=camera_id,
                camera_position=camera_position,
                image=image,
                width=first_image.get("width", image.shape[1]),
                height=first_image.get("height", image.shape[0]),
                pixel_format=first_image.get("pixel_format", "BGR"),
                sequence_id=sequence_id,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse message: {e}", exc_info=True)
            return None

    def _fetch_image(self, image_url: str) -> Optional[bytes]:
        try:
            import requests
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.warning(f"Failed to fetch image from {image_url}: {e}")
        return None

    def stop(self):
        if not self._is_running:
            return

        logger.info("Stopping message consumer...")
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        if self._type == "rabbitmq" and hasattr(self, '_connection'):
            try:
                if self._connection.is_open:
                    self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing RabbitMQ connection: {e}")

        elif self._type == "kafka" and self._consumer:
            try:
                self._consumer.close()
            except Exception as e:
                logger.warning(f"Error closing Kafka consumer: {e}")

        self._is_running = False
        logger.info("Message consumer stopped")

    def is_connected(self) -> bool:
        if self._type == "rabbitmq":
            return hasattr(self, '_connection') and self._connection.is_open
        elif self._type == "kafka":
            return self._consumer is not None
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "is_running": self._is_running,
            "is_connected": self.is_connected(),
            "queue_type": self._type
        }

    def reconnect(self) -> bool:
        self.stop()
        time.sleep(1)
        return self.connect()
