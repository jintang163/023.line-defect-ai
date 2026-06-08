from typing import Dict, Any, Optional
import time
import json
import base64
import numpy as np
import cv2
import threading

from src.utils.schemas import DetectionOutput
from src.utils.logger import Logger

logger = Logger().logger

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


class ResultProducer:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._connection = None
        self._channel = None
        self._producer = None
        self._is_connected = False
        self._lock = threading.Lock()

        self._type = config.get("type", "rabbitmq")
        self._host = config.get("host", "localhost")
        self._port = config.get("port", 5672)
        self._username = config.get("username", "guest")
        self._password = config.get("password", "guest")
        self._exchange = config.get("exchange", "line-defect-exchange")
        self._routing_key = config.get("routing_key", "defect.result")
        self._queue = config.get("queue", "defect-result-queue")
        self._topic = config.get("topic", "defect-result-topic")

        self._stats = {
            "total_sent": 0,
            "total_errors": 0
        }

    def connect(self) -> bool:
        with self._lock:
            try:
                if self._type == "rabbitmq":
                    return self._connect_rabbitmq()
                elif self._type == "kafka":
                    return self._connect_kafka()
                else:
                    logger.error(f"Unsupported message queue type: {self._type}")
                    return False
            except Exception as e:
                logger.error(f"Failed to connect result producer: {e}", exc_info=True)
                self._is_connected = False
                return False

    def _connect_rabbitmq(self) -> bool:
        if not RABBITMQ_AVAILABLE:
            logger.error("RabbitMQ not available")
            return False

        try:
            credentials = pika.PlainCredentials(self._username, self._password)
            parameters = pika.ConnectionParameters(
                host=self._host,
                port=self._port,
                credentials=credentials,
                heartbeat=600
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
                durable=True
            )

            self._channel.queue_bind(
                exchange=self._exchange,
                queue=self._queue,
                routing_key=self._routing_key
            )

            self._is_connected = True
            logger.info(f"Result producer connected to RabbitMQ at {self._host}:{self._port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect RabbitMQ producer: {e}")
            return False

    def _connect_kafka(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available")
            return False

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=f"{self._host}:{self._port}",
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )

            self._is_connected = True
            logger.info(f"Result producer connected to Kafka at {self._host}:{self._port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect Kafka producer: {e}")
            return False

    def send_result(self, detection_output: DetectionOutput,
                    annotated_image: Optional[np.ndarray] = None) -> bool:
        try:
            message = self._build_message(detection_output, annotated_image)

            with self._lock:
                if not self._is_connected:
                    if not self.connect():
                        logger.error("Cannot send result: not connected")
                        self._stats["total_errors"] += 1
                        return False

                if self._type == "rabbitmq":
                    return self._send_rabbitmq(message)
                elif self._type == "kafka":
                    return self._send_kafka(message)

            return False

        except Exception as e:
            logger.error(f"Failed to send result: {e}", exc_info=True)
            self._stats["total_errors"] += 1
            return False

    def _build_message(self, detection_output: DetectionOutput,
                       annotated_image: Optional[np.ndarray]) -> Dict[str, Any]:
        message = detection_output.to_dict()

        message["defects"] = [d.to_dict() for d in detection_output.defects]

        if annotated_image is not None:
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, img_encoded = cv2.imencode('.jpg', annotated_image, encode_param)
                image_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                message["annotated_image"] = image_base64
            except Exception as e:
                logger.warning(f"Failed to encode annotated image: {e}")

        if detection_output.image_data:
            message["image_info"] = {
                "image_id": detection_output.image_data.image_id,
                "camera_id": detection_output.image_data.camera_id,
                "camera_position": detection_output.image_data.camera_position,
                "width": detection_output.image_data.width,
                "height": detection_output.image_data.height
            }

        message["summary"] = {
            "critical_defects": len(detection_output.critical_defects),
            "major_defects": len(detection_output.major_defects),
            "minor_defects": len(detection_output.minor_defects),
            "total_defects": len(detection_output.defects)
        }

        return message

    def _send_rabbitmq(self, message: Dict[str, Any]) -> bool:
        try:
            message_body = json.dumps(message, ensure_ascii=False).encode('utf-8')

            properties = pika.BasicProperties(
                delivery_mode=2,
                content_type='application/json',
                timestamp=int(time.time())
            )

            self._channel.basic_publish(
                exchange=self._exchange,
                routing_key=self._routing_key,
                body=message_body,
                properties=properties
            )

            self._stats["total_sent"] += 1
            logger.debug(f"Sent result for detection_id: {message.get('detection_id')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send RabbitMQ message: {e}")
            self._is_connected = False
            return False

    def _send_kafka(self, message: Dict[str, Any]) -> bool:
        try:
            key = message.get("product_id", "default")
            future = self._producer.send(
                self._topic,
                value=message,
                key=key
            )
            future.get(timeout=10)

            self._stats["total_sent"] += 1
            logger.debug(f"Sent result for detection_id: {message.get('detection_id')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Kafka message: {e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "is_connected": self._is_connected,
            "queue_type": self._type
        }

    def reconnect(self) -> bool:
        self.disconnect()
        time.sleep(1)
        return self.connect()

    def disconnect(self):
        with self._lock:
            if self._type == "rabbitmq" and self._connection:
                try:
                    if self._connection.is_open:
                        self._connection.close()
                except Exception as e:
                    logger.warning(f"Error closing RabbitMQ connection: {e}")
                self._connection = None
                self._channel = None

            elif self._type == "kafka" and self._producer:
                try:
                    self._producer.flush(timeout=5)
                    self._producer.close()
                except Exception as e:
                    logger.warning(f"Error closing Kafka producer: {e}")
                self._producer = None

            self._is_connected = False
            logger.info("Result producer disconnected")
