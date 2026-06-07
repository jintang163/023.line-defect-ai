import threading
import time
from collections import deque
from typing import Optional, Callable, List, Any
from src.utils.logger import Logger

logger = Logger().logger


class RingBuffer:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._overflow_count = 0

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        with self._not_empty:
            if len(self._buffer) >= self.max_size:
                if block:
                    start = time.time()
                    while len(self._buffer) >= self.max_size:
                        remaining = timeout - (time.time() - start) if timeout else None
                        if remaining is not None and remaining <= 0:
                            self._overflow_count += 1
                            logger.warning(f"Ring buffer overflow, count: {self._overflow_count}")
                            return False
                        self._not_empty.wait(remaining)
                else:
                    self._overflow_count += 1
                    logger.warning(f"Ring buffer overflow, count: {self._overflow_count}")
                    return False

            self._buffer.append(item)
            self._not_empty.notify_all()
            return True

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        with self._not_empty:
            if not self._buffer:
                if block:
                    self._not_empty.wait(timeout)
                    if not self._buffer:
                        return None
                else:
                    return None

            item = self._buffer.popleft()
            self._not_empty.notify_all()
            return item

    def get_all(self, max_count: int = 10) -> List[Any]:
        with self._lock:
            count = min(max_count, len(self._buffer))
            items = [self._buffer.popleft() for _ in range(count)]
            return items

    def peek(self) -> Optional[Any]:
        with self._lock:
            return self._buffer[0] if self._buffer else None

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._not_empty.notify_all()

    def qsize(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self.max_size

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    def __len__(self) -> int:
        return self.qsize()
