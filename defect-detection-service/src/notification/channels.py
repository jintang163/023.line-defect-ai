from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import threading

from src.utils.logger import Logger

logger = Logger("notification_channels", "INFO", "./logs/defect-detection.log").logger

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    SMTP_AVAILABLE = True
except ImportError:
    SMTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class NotificationChannel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @abstractmethod
    def send(self, recipients: List[str], subject: str, content: str,
             alert_level: str = "warning", details: Optional[Dict[str, Any]] = None) -> bool:
        pass

    @abstractmethod
    def channel_name(self) -> str:
        pass


class EmailChannel(NotificationChannel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._smtp_server = config.get("smtp_server", "")
        self._smtp_port = config.get("smtp_port", 587)
        self._smtp_username = config.get("smtp_username", "")
        self._smtp_password = config.get("smtp_password", "")
        self._sender = config.get("sender", self._smtp_username)
        self._use_tls = config.get("use_tls", True)

    def channel_name(self) -> str:
        return "email"

    def send(self, recipients: List[str], subject: str, content: str,
             alert_level: str = "warning", details: Optional[Dict[str, Any]] = None) -> bool:
        if not self._enabled or not SMTP_AVAILABLE:
            return False
        if not recipients:
            return False

        try:
            level_emoji = {
                "critical": "🔴",
                "error": "🟠",
                "warning": "🟡",
                "info": "🔵"
            }.get(alert_level, "⚪")

            html_content = f"""
<html><body style="font-family:Arial,sans-serif;font-size:14px;color:#333">
<div style="border-left:4px solid {'#e74c3c' if alert_level == 'critical' else '#f39c12' if alert_level in ('error','warning') else '#3498db'};padding:12px 16px;margin-bottom:16px;background:#f9f9f9">
<h3 style="margin:0 0 8px">{level_emoji} 缺陷检测告警 - {alert_level.upper()}</h3>
<p style="margin:4px 0"><strong>主题：</strong>{subject}</p>
<p style="margin:4px 0"><strong>级别：</strong>{alert_level}</p>
<p style="margin:4px 0"><strong>时间：</strong>{content[:200]}</p>
</div>
<div style="padding:8px 16px;background:#fff;border:1px solid #ddd">
<h4 style="margin:8px 0">详细信息</h4>
<pre style="white-space:pre-wrap;font-size:13px">{content}</pre>
</div>
</body></html>"""

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{level_emoji} [缺陷检测告警] {subject}"
            msg["From"] = self._sender
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(html_content, "html", "utf-8"))

            with smtplib.SMTP(self._smtp_server, self._smtp_port, timeout=10) as server:
                if self._use_tls:
                    server.starttls()
                if self._smtp_username and self._smtp_password:
                    server.login(self._smtp_username, self._smtp_password)
                server.sendmail(self._sender, recipients, msg.as_string())

            logger.info(f"Email sent to {recipients}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {recipients}: {e}", exc_info=True)
            return False


class DingTalkChannel(NotificationChannel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._webhook_url = config.get("webhook_url", "")
        self._secret = config.get("secret", "")
        self._at_mobiles = config.get("at_mobiles", [])
        self._at_all = config.get("at_all", False)

    def channel_name(self) -> str:
        return "dingtalk"

    def send(self, recipients: List[str], subject: str, content: str,
             alert_level: str = "warning", details: Optional[Dict[str, Any]] = None) -> bool:
        if not self._enabled or not REQUESTS_AVAILABLE or not self._webhook_url:
            return False

        try:
            import hashlib
            import hmac
            import base64
            import urllib.parse
            import time as _time

            url = self._webhook_url
            if self._secret:
                timestamp = str(round(_time.time() * 1000))
                string_to_sign = f"{timestamp}\n{self._secret}"
                hmac_code = hmac.new(
                    self._secret.encode("utf-8"),
                    string_to_sign.encode("utf-8"),
                    digestmod=hashlib.sha256
                ).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                url = f"{self._webhook_url}&timestamp={timestamp}&sign={sign}"

            level_color = {
                "critical": "#FF0000",
                "error": "#FF6600",
                "warning": "#FFCC00",
                "info": "#0088CC"
            }.get(alert_level, "#999999")

            at_mobiles = self._at_mobiles if self._at_all else []
            at_text = " @所有人" if self._at_all else ""

            markdown_text = (
                f"### 缺陷检测告警 {alert_level.upper()}\n\n"
                f"> **主题：** {subject}\n\n"
                f"> **级别：** <font color={level_color}>{alert_level}</font>\n\n"
                f"> **内容：** {content[:500]}\n\n"
                f"{at_text}"
            )

            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": f"缺陷检测告警 - {subject}",
                    "text": markdown_text
                },
                "at": {
                    "atMobiles": at_mobiles,
                    "isAtAll": self._at_all
                }
            }

            resp = requests.post(
                url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )

            if resp.status_code == 200:
                result = resp.json()
                if result.get("errcode") == 0:
                    logger.info(f"DingTalk notification sent: {subject}")
                    return True
                else:
                    logger.warning(f"DingTalk API error: {result}")
                    return False
            else:
                logger.warning(f"DingTalk HTTP error: {resp.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send DingTalk notification: {e}", exc_info=True)
            return False


class SmsChannel(NotificationChannel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._provider = config.get("provider", "")
        self._api_url = config.get("api_url", "")
        self._api_key = config.get("api_key", "")
        self._api_secret = config.get("api_secret", "")
        self._sign_name = config.get("sign_name", "")
        self._template_code = config.get("template_code", "")

    def channel_name(self) -> str:
        return "sms"

    def send(self, recipients: List[str], subject: str, content: str,
             alert_level: str = "warning", details: Optional[Dict[str, Any]] = None) -> bool:
        if not self._enabled or not REQUESTS_AVAILABLE or not self._api_url:
            return False
        if not recipients:
            return False

        try:
            level_text = {
                "critical": "紧急",
                "error": "重要",
                "warning": "一般",
                "info": "提示"
            }.get(alert_level, "告警")

            sms_content = f"【{self._sign_name}】{level_text}告警：{subject} - {content[:100]}"

            for phone in recipients:
                payload = {
                    "phone": phone,
                    "content": sms_content,
                    "api_key": self._api_key,
                    "api_secret": self._api_secret,
                    "sign_name": self._sign_name,
                    "template_code": self._template_code,
                    "template_params": {
                        "level": level_text,
                        "subject": subject[:20],
                        "content": content[:50]
                    }
                }

                resp = requests.post(
                    self._api_url,
                    json=payload,
                    timeout=10,
                    headers={"Content-Type": "application/json"}
                )

                if resp.status_code != 200:
                    logger.warning(f"SMS send failed for {phone}: HTTP {resp.status_code}")
                    return False

            logger.info(f"SMS sent to {recipients}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS to {recipients}: {e}", exc_info=True)
            return False
