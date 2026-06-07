import json
import time
from typing import Dict, Any, Optional
import pika
from pika.exceptions import AMQPConnectionError, AMQPError
from src.messaging.base_producer import BaseMessageProducer
from src.utils.schemas import ImageMessage
from src.utils.logger import Logger

logger = Logger().logger


class RabbitMQProducer(BaseMessageProducer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        rabbit_cfg = config.get("rabbitmq", {})
        self.host = rabbit_cfg.get("host", "localhost")
        self.port = rabbit_cfg.get("port", 5672)
        self.username = rabbit_cfg.get("username", "admin")
        self.password = rabbit_cfg.get("password", "admin")
        self.virtual_host = rabbit_cfg.get("virtual_host", "/")
        self.exchange = rabbit_cfg.get("exchange", "defect.images")
        self.routing_key = rabbit_cfg.get("routing_key", "image.raw")
        self.queue = rabbit_cfg.get("queue", "defect.image.queue")
        self.exchange_type = rabbit_cfg.get("exchange_type", "direct")
        self.durable = rabbit_cfg.get("durable", True)

        self._connection = None
        self._channel = None
        self._last_reconnect = 0
        self._reconnect_interval = 5

    def connect(self) -> bool:
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )

            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()

            self._channel.exchange_declare(
                exchange=self.exchange,
                exchange_type=self.exchange_type,
                durable=self.durable
            )

            self._channel.queue_declare(
                queue=self.queue,
                durable=self.durable,
                arguments={"x-max-priority": 10}
            )

            self._channel.queue_bind(
                queue=self.queue,
                exchange=self.exchange,
                routing_key=self.routing_key
            )

            self._is_connected = True
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
            return True

        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to RabbitMQ: {e}")
            self._is_connected = False
            return False

    def _reconnect_if_needed(self) -> bool:
        if self._is_connected and self._connection and self._connection.is_open:
            return True

        now = time.time()
        if now - self._last_reconnect < self._reconnect_interval:
            return False

        logger.info("Attempting to reconnect to RabbitMQ...")
        self._last_reconnect = now
        return self.connect()

    def disconnect(self):
        try:
            if self._channel:
                self._channel.close()
            if self._connection:
                self._connection.close()
        except Exception as e:
            logger.error(f"Error disconnecting RabbitMQ: {e}")
        finally:
            self._channel = None
            self._connection = None
            self._is_connected = False
            logger.info("Disconnected from RabbitMQ")

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

            properties = pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                timestamp=int(time.time()),
                message_id=message.sequence_id
            )

            self._channel.basic_publish(
                exchange=self.exchange,
                routing_key=self.routing_key,
                body=body,
                properties=properties
            )

            self._increment_send_count()
            logger.debug(f"Sent message {message.sequence_id} to RabbitMQ")
            return True

        except (AMQPError, AMQPConnectionError) as e:
            logger.error(f"RabbitMQ send error: {e}")
            self._is_connected = False
            self._increment_fail_count()
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to RabbitMQ: {e}")
            self._increment_fail_count()
            return False

    def send_raw(self, body: bytes, message_id: str) -> bool:
        if not self._reconnect_if_needed():
            self._increment_fail_count()
            return False

        try:
            properties = pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                timestamp=int(time.time()),
                message_id=message_id
            )

            self._channel.basic_publish(
                exchange=self.exchange,
                routing_key=self.routing_key,
                body=body,
                properties=properties
            )

            self._increment_send_count()
            return True

        except Exception as e:
            logger.error(f"RabbitMQ raw send error: {e}")
            self._is_connected = False
            self._increment_fail_count()
            return False
