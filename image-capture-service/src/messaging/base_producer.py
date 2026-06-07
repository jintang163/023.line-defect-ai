from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.utils.schemas import ImageMessage
from src.utils.logger import Logger

logger = Logger().logger


class BaseMessageProducer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_connected = False
        self._send_count = 0
        self._fail_count = 0

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def send(self, message: ImageMessage, images_data: Optional[Dict[str, bytes]] = None) -> bool:
        pass

    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def send_count(self) -> int:
        return self._send_count

    @property
    def fail_count(self) -> int:
        return self._fail_count

    def _increment_send_count(self):
        self._send_count += 1

    def _increment_fail_count(self):
        self._fail_count += 1
