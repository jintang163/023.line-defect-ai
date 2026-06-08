from typing import Optional, Callable, Dict, Any, List
import threading
import time
import json
import numpy as np
import cv2
import base64
from urllib.parse import urlparse

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

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available, URL image fetching disabled")

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("minio not available, MinIO integration disabled")


class MessageConsumer:
    CONFIG_KEY_MAPPING = {
        "type": ["type", "mq_type", "broker_type"],
        "host": ["host", "mq_host", "broker_host", "server"],
        "port": ["port", "mq_port", "broker_port"],
        "username": ["username", "mq_username", "user", "user_name"],
        "password": ["password", "mq_password", "passwd", "pwd"],
        "queue": ["queue", "input_queue", "queue_name", "image_queue"],
        "topic": ["topic", "input_topic", "kafka_topic"],
        "exchange": ["exchange", "mq_exchange", "exchange_name"],
        "routing_key": ["routing_key", "input_routing_key", "binding_key"],
    }

    def __init__(self, config: Dict[str, Any],
                 callback: Optional[Callable[[ImageData], None]] = None,
                 product_switch_callback: Optional[Callable[[str], bool]] = None):
        self._config = config
        self._callback = callback
        self._product_switch_callback = product_switch_callback
        self._consumer = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._session = None
        self._minio_client = None

        self._type = self._get_config("type", "rabbitmq")
        self._host = self._get_config("host", "localhost")
        self._port = self._get_config("port", 5672)
        self._username = self._get_config("username", "guest")
        self._password = self._get_config("password", "guest")
        self._queue = self._get_config("queue", "image-capture-queue")
        self._topic = self._get_config("topic", "image-capture-topic")
        self._exchange = self._get_config("exchange", "line-defect-exchange")
        self._routing_key = self._get_config("routing_key", "image.captured")

        self._image_fetch_timeout = config.get("image_fetch_timeout", 10)
        self._image_fetch_retries = config.get("image_fetch_retries", 3)
        self._image_fetch_retry_delay = config.get("image_fetch_retry_delay", 1)

        self._process_all_cameras = config.get("process_all_cameras", True)
        self._camera_position_filter = config.get("camera_position_filter", [])
        self._camera_id_filter = config.get("camera_id_filter", [])

        self._auto_switch_product = config.get("auto_switch_product", True)
        self._current_product_id: Optional[str] = None

        self._init_minio_client(config)

        self._stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_errors": 0,
            "total_images": 0,
            "image_fetch_errors": 0,
            "product_switches": 0,
            "minio_fetch_count": 0,
            "url_fetch_count": 0,
            "inline_fetch_count": 0,
        }

    def _get_config(self, key: str, default: Any) -> Any:
        possible_keys = self.CONFIG_KEY_MAPPING.get(key, [key])
        for k in possible_keys:
            if k in self._config:
                value = self._config[k]
                logger.debug(f"配置键映射: {k} => {value}")
                return value
        return default

    def _init_minio_client(self, config: Dict[str, Any]):
        if not MINIO_AVAILABLE:
            return

        image_storage = config.get("image_storage", {})
        if image_storage.get("type") != "minio":
            return

        try:
            endpoint = image_storage.get("endpoint", "localhost:9000")
            access_key = image_storage.get("access_key", "minioadmin")
            secret_key = image_storage.get("secret_key", "minioadmin")
            secure = image_storage.get("use_ssl", False)

            self._minio_client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )

            self._minio_bucket = image_storage.get("bucket", "defect-images")
            logger.info(f"✅ MinIO 客户端已初始化，Endpoint: {endpoint}, Bucket: {self._minio_bucket}")

        except Exception as e:
            logger.warning(f"⚠️  MinIO 客户端初始化失败: {e}")
            self._minio_client = None

    def set_callback(self, callback: Callable[[ImageData], None]):
        self._callback = callback

    def set_product_switch_callback(self, callback: Callable[[str], bool]):
        self._product_switch_callback = callback

    def connect(self) -> bool:
        try:
            if self._type == "rabbitmq":
                return self._connect_rabbitmq()
            elif self._type == "kafka":
                return self._connect_kafka()
            else:
                logger.error(f"❌ 不支持的消息队列类型: {self._type}")
                return False
        except Exception as e:
            logger.error(f"❌ 连接消息队列失败: {e}", exc_info=True)
            return False

    def _connect_rabbitmq(self) -> bool:
        if not RABBITMQ_AVAILABLE:
            logger.error("❌ RabbitMQ 不可用，请安装 pika")
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

            logger.info(f"✅ 已连接到 RabbitMQ: {self._host}:{self._port}, Queue: {self._queue}")
            return True

        except Exception as e:
            logger.error(f"❌ 连接 RabbitMQ 失败: {e}", exc_info=True)
            return False

    def _connect_kafka(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.error("❌ Kafka 不可用，请安装 kafka-python")
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

            logger.info(f"✅ 已连接到 Kafka: {self._host}:{self._port}, Topic: {self._topic}")
            return True

        except Exception as e:
            logger.error(f"❌ 连接 Kafka 失败: {e}", exc_info=True)
            return False

    def start(self):
        if self._is_running:
            logger.warning("⚠️  消费者已经在运行")
            return

        if not self.connect():
            logger.error("❌ 启动消费者失败：连接失败")
            return

        if REQUESTS_AVAILABLE:
            self._session = requests.Session()

        self._is_running = True
        self._stop_event.clear()

        if self._type == "rabbitmq":
            self._thread = threading.Thread(target=self._run_rabbitmq, daemon=True)
        elif self._type == "kafka":
            self._thread = threading.Thread(target=self._run_kafka, daemon=True)

        self._thread.start()
        logger.info("🚀 消息消费者已启动")
        logger.info(f"   处理所有相机: {self._process_all_cameras}")
        logger.info(f"   自动切换产品: {self._auto_switch_product}")

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
            logger.error(f"❌ RabbitMQ 消费者错误: {e}", exc_info=True)
        finally:
            self._is_running = False
            logger.info("⏹️  RabbitMQ 消费者已停止")

    def _rabbitmq_callback(self, ch, method, properties, body):
        try:
            message = json.loads(body.decode('utf-8'))
            self._stats["total_received"] += 1

            image_data_list = self._parse_message(message)
            if image_data_list:
                for image_data in image_data_list:
                    if self._callback:
                        self._callback(image_data)
                self._stats["total_processed"] += 1
                self._stats["total_images"] += len(image_data_list)

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"❌ 处理 RabbitMQ 消息错误: {e}", exc_info=True)
            self._stats["total_errors"] += 1
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _run_kafka(self):
        try:
            for message in self._consumer:
                if self._stop_event.is_set():
                    break

                try:
                    self._stats["total_received"] += 1

                    image_data_list = self._parse_message(message.value)
                    if image_data_list:
                        for image_data in image_data_list:
                            if self._callback:
                                self._callback(image_data)
                        self._stats["total_processed"] += 1
                        self._stats["total_images"] += len(image_data_list)

                except Exception as e:
                    logger.error(f"❌ 处理 Kafka 消息错误: {e}", exc_info=True)
                    self._stats["total_errors"] += 1

        except Exception as e:
            logger.error(f"❌ Kafka 消费者错误: {e}", exc_info=True)
        finally:
            self._is_running = False
            logger.info("⏹️  Kafka 消费者已停止")

    def _parse_message(self, message: Dict[str, Any]) -> Optional[List[ImageData]]:
        try:
            sequence_id = message.get("sequence_id", str(int(time.time() * 1000)))
            line_id = message.get("line_id", message.get("line", ""))
            product_id = message.get("product_id", message.get("product", ""))

            if self._auto_switch_product and product_id and product_id != self._current_product_id:
                if self._product_switch_callback:
                    success = self._product_switch_callback(product_id)
                    if success:
                        self._current_product_id = product_id
                        self._stats["product_switches"] += 1
                        logger.info(f"🔄 自动切换到产品: {product_id}")

            images_info = message.get("images", [])
            image_data_map = message.get("image_data", message.get("images_data", {}))
            image_urls_map = message.get("image_urls", message.get("urls", {}))

            if not images_info:
                logger.warning(f"⚠️  消息 {sequence_id} 中没有图像信息")
                return None

            result_images: List[ImageData] = []

            for image_info in images_info:
                camera_id = image_info.get("camera_id", image_info.get("camera", ""))
                camera_position = image_info.get("camera_position", image_info.get("position", ""))

                if self._camera_position_filter and camera_position not in self._camera_position_filter:
                    logger.debug(f"⏭️  跳过相机 {camera_id} (位置过滤: {camera_position})")
                    continue

                if self._camera_id_filter and camera_id not in self._camera_id_filter:
                    logger.debug(f"⏭️  跳过相机 {camera_id} (ID过滤)")
                    continue

                image_id = image_info.get("image_id", "")
                timestamp = image_info.get("timestamp", time.time())
                width = image_info.get("width", 0)
                height = image_info.get("height", 0)
                pixel_format = image_info.get("pixel_format", "BGR")
                metadata = image_info.get("metadata", {})

                image_bytes = self._get_image_bytes(
                    image_id=image_id,
                    image_data_map=image_data_map,
                    image_urls_map=image_urls_map,
                    image_info=image_info
                )

                if image_bytes is None:
                    logger.warning(f"⚠️  无法获取图像 {image_id} 的数据")
                    self._stats["image_fetch_errors"] += 1
                    continue

                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    logger.error(f"❌ 无法解码图像: {image_id}")
                    self._stats["image_fetch_errors"] += 1
                    continue

                actual_width = width if width > 0 else image.shape[1]
                actual_height = height if height > 0 else image.shape[0]

                img_metadata = {
                    "line_id": line_id,
                    "product_id": product_id,
                    "original_timestamp": timestamp,
                    "camera_metadata": metadata,
                    "sequence_id": sequence_id
                }

                image_data = ImageData.create(
                    camera_id=camera_id,
                    camera_position=camera_position,
                    image=image,
                    width=actual_width,
                    height=actual_height,
                    pixel_format=pixel_format,
                    sequence_id=sequence_id,
                    metadata=img_metadata
                )

                result_images.append(image_data)

                if not self._process_all_cameras:
                    break

            return result_images if result_images else None

        except Exception as e:
            logger.error(f"❌ 解析消息失败: {e}", exc_info=True)
            return None

    def _get_image_bytes(self, image_id: str, image_data_map: Dict[str, Any],
                         image_urls_map: Dict[str, Any], image_info: Dict[str, Any]) -> Optional[bytes]:
        image_bytes = None

        if image_id in image_data_map:
            image_bytes = image_data_map[image_id]
            if isinstance(image_bytes, str):
                try:
                    image_bytes = base64.b64decode(image_bytes)
                    self._stats["inline_fetch_count"] += 1
                    logger.debug(f"📦 使用内嵌图像数据: {image_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Base64 解码失败: {e}")
                    image_bytes = None

        if image_bytes is None:
            if image_id in image_urls_map:
                image_url = image_urls_map[image_id]
                image_bytes = self._fetch_image_with_retry(image_url, image_id)

        if image_bytes is None:
            image_url = image_info.get("image_url", image_info.get("url", ""))
            if image_url:
                image_bytes = self._fetch_image_with_retry(image_url, image_id)

        if image_bytes is None and self._minio_client:
            image_bytes = self._fetch_from_minio(image_id)

        return image_bytes

    def _fetch_image_with_retry(self, image_url: str, image_id: str) -> Optional[bytes]:
        if not REQUESTS_AVAILABLE or self._session is None:
            return None

        for attempt in range(1, self._image_fetch_retries + 1):
            try:
                logger.debug(f"🌐 拉取图像 [{attempt}/{self._image_fetch_retries}]: {image_url}")

                response = self._session.get(
                    image_url,
                    timeout=self._image_fetch_timeout
                )

                if response.status_code == 200:
                    self._stats["url_fetch_count"] += 1
                    logger.debug(f"✅ URL 拉取成功: {image_id}")
                    return response.content
                elif response.status_code == 404:
                    logger.warning(f"⚠️  图像不存在 (404): {image_url}")
                    break
                else:
                    logger.warning(f"⚠️  URL 拉取失败 [{attempt}/{self._image_fetch_retries}]: "
                                   f"HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"⚠️  URL 拉取超时 [{attempt}/{self._image_fetch_retries}]: {image_url}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"⚠️  连接错误 [{attempt}/{self._image_fetch_retries}]: {e}")
            except Exception as e:
                logger.warning(f"⚠️  URL 拉取异常 [{attempt}/{self._image_fetch_retries}]: {e}")

            if attempt < self._image_fetch_retries:
                time.sleep(self._image_fetch_retry_delay)

        logger.error(f"❌ 图像 URL 拉取最终失败: {image_url}")
        return None

    def _fetch_from_minio(self, image_id: str) -> Optional[bytes]:
        if not self._minio_client or not self._minio_bucket:
            return None

        for attempt in range(1, self._image_fetch_retries + 1):
            try:
                object_name = image_id
                if not object_name.endswith('.jpg') and not object_name.endswith('.jpeg'):
                    object_name = f"{image_id}.jpg"

                logger.debug(f"☁️  从 MinIO 拉取 [{attempt}/{self._image_fetch_retries}]: {object_name}")

                response = self._minio_client.get_object(
                    self._minio_bucket,
                    object_name
                )

                data = response.read()
                response.close()
                response.release_conn()

                if data and len(data) > 0:
                    self._stats["minio_fetch_count"] += 1
                    logger.debug(f"✅ MinIO 拉取成功: {image_id}")
                    return data

            except S3Error as e:
                if e.code == "NoSuchKey":
                    logger.warning(f"⚠️  MinIO 对象不存在: {image_id}")
                    break
                logger.warning(f"⚠️  MinIO 拉取错误 [{attempt}/{self._image_fetch_retries}]: {e}")
            except Exception as e:
                logger.warning(f"⚠️  MinIO 拉取异常 [{attempt}/{self._image_fetch_retries}]: {e}")

            if attempt < self._image_fetch_retries:
                time.sleep(self._image_fetch_retry_delay)

        return None

    def stop(self):
        if not self._is_running:
            return

        logger.info("⏹️  正在停止消息消费者...")
        self._stop_event.set()

        if self._session:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"⚠️  关闭 HTTP Session 错误: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        if self._type == "rabbitmq" and hasattr(self, '_connection'):
            try:
                if self._connection.is_open:
                    self._connection.close()
            except Exception as e:
                logger.warning(f"⚠️  关闭 RabbitMQ 连接错误: {e}")

        elif self._type == "kafka" and self._consumer:
            try:
                self._consumer.close()
            except Exception as e:
                logger.warning(f"⚠️  关闭 Kafka 消费者错误: {e}")

        self._is_running = False
        logger.info("⏹️  消息消费者已停止")

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
            "queue_type": self._type,
            "current_product_id": self._current_product_id,
            "minio_available": self._minio_client is not None
        }

    def reconnect(self) -> bool:
        self.stop()
        time.sleep(1)
        return self.connect()
