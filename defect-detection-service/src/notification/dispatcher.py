from typing import Dict, Any, List, Optional
import threading
import time
from datetime import datetime

from src.utils.logger import Logger
from src.notification.channels import (
    NotificationChannel, EmailChannel, DingTalkChannel, SmsChannel
)

logger = Logger("notification_dispatcher", "INFO", "./logs/defect-detection.log").logger


class NotificationDispatcher:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)
        self._channels: Dict[str, NotificationChannel] = {}
        self._receiver_groups: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._send_history: List[Dict[str, Any]] = []
        self._max_history = config.get("max_history", 5000)

        if self._enabled:
            self._init_channels()
            self._init_receiver_groups()

    def _init_channels(self):
        channels_config = self._config.get("channels", {})

        email_cfg = channels_config.get("email", {})
        if email_cfg.get("enable", False):
            self._channels["email"] = EmailChannel(email_cfg)
            logger.info("Email notification channel enabled")

        dingtalk_cfg = channels_config.get("dingtalk", {})
        if dingtalk_cfg.get("enable", False):
            self._channels["dingtalk"] = DingTalkChannel(dingtalk_cfg)
            logger.info("DingTalk notification channel enabled")

        sms_cfg = channels_config.get("sms", {})
        if sms_cfg.get("enable", False):
            self._channels["sms"] = SmsChannel(sms_cfg)
            logger.info("SMS notification channel enabled")

    def _init_receiver_groups(self):
        self._receiver_groups = self._config.get("receiver_groups", {})
        if not self._receiver_groups:
            self._receiver_groups = {
                "urgent": {
                    "channels": ["email", "dingtalk", "sms"],
                    "receivers": {
                        "email": [],
                        "dingtalk": [],
                        "sms": []
                    }
                },
                "warning": {
                    "channels": ["email", "dingtalk"],
                    "receivers": {
                        "email": [],
                        "dingtalk": []
                    }
                },
                "info": {
                    "channels": ["dingtalk"],
                    "receivers": {
                        "dingtalk": []
                    }
                }
            }

    def dispatch(self, alert_level: str, category: str, subject: str,
                 content: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        if not self._enabled:
            return {}

        results: Dict[str, bool] = {}

        group_name = self._map_level_to_group(alert_level)
        group = self._receiver_groups.get(group_name, {})

        channels_to_use = group.get("channels", [])
        receivers_by_channel = group.get("receivers", {})

        if not channels_to_use:
            default_channels = group.get("channels", ["dingtalk"])
            channels_to_use = default_channels

        for channel_name in channels_to_use:
            channel = self._channels.get(channel_name)
            if not channel or not channel.enabled:
                results[channel_name] = False
                continue

            recipients = receivers_by_channel.get(channel_name, [])
            if not recipients:
                results[channel_name] = False
                continue

            try:
                success = channel.send(
                    recipients=recipients,
                    subject=subject,
                    content=content,
                    alert_level=alert_level,
                    details=details
                )
                results[channel_name] = success
            except Exception as e:
                logger.error(f"Error dispatching via {channel_name}: {e}", exc_info=True)
                results[channel_name] = False

        self._record_dispatch(alert_level, category, subject, channels_to_use, results)

        return results

    def _map_level_to_group(self, alert_level: str) -> str:
        mapping = {
            "critical": "urgent",
            "error": "urgent",
            "warning": "warning",
            "info": "info"
        }
        return mapping.get(alert_level, "warning")

    def _record_dispatch(self, alert_level: str, category: str, subject: str,
                         channels: List[str], results: Dict[str, bool]):
        with self._lock:
            self._send_history.append({
                "timestamp": time.time(),
                "timestamp_iso": datetime.now().isoformat(),
                "alert_level": alert_level,
                "category": category,
                "subject": subject,
                "channels": channels,
                "results": results
            })
            if len(self._send_history) > self._max_history:
                self._send_history = self._send_history[-self._max_history:]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._send_history)
            success = sum(1 for h in self._send_history if any(v for v in h.get("results", {}).values()))
            return {
                "enabled": self._enabled,
                "channels": {name: ch.enabled for name, ch in self._channels.items()},
                "receiver_groups": list(self._receiver_groups.keys()),
                "total_dispatched": total,
                "total_success": success,
                "total_failed": total - success
            }

    def get_send_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return self._send_history[-limit:]

    def update_receivers(self, group_name: str, channel_name: str, receivers: List[str]):
        with self._lock:
            if group_name not in self._receiver_groups:
                self._receiver_groups[group_name] = {
                    "channels": [channel_name],
                    "receivers": {channel_name: receivers}
                }
            else:
                group = self._receiver_groups[group_name]
                group.setdefault("receivers", {})[channel_name] = receivers
                if channel_name not in group.get("channels", []):
                    group.setdefault("channels", []).append(channel_name)

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "channels": {name: {"enabled": ch.enabled, "name": ch.channel_name()}
                             for name, ch in self._channels.items()},
                "receiver_groups": self._receiver_groups
            }

    @property
    def enabled(self) -> bool:
        return self._enabled
