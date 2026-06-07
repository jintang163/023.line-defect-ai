import logging
import coloredlogs
import sys
from datetime import datetime
from typing import Optional
import os


class Logger:
    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "capture-service", level: str = "INFO", log_file: Optional[str] = None):
        if self._logger is not None:
            return

        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))

        log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(process)d | %(filename)s:%(lineno)d | %(message)s"
        coloredlogs.install(
            level=level.upper(),
            logger=self._logger,
            fmt=log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(logging.Formatter(log_format))
            self._logger.addHandler(file_handler)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
