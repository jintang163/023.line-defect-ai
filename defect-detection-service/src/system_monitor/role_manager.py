import hmac
import hashlib
import base64
import json
import time
from enum import Enum
from typing import Optional, Dict, List

from src.utils.logger import Logger

logger = Logger("role_manager", "INFO", "./logs/defect-detection.log").logger


class RoleManager:
    class Role(Enum):
        OPERATOR = "operator"
        ENGINEER = "engineer"
        ADMIN = "admin"

    class Permission(Enum):
        VIEW = "view"
        MANUAL_OVERRIDE = "manual_override"
        ADJUST_PARAMS = "adjust_params"
        FULL_CONFIG = "full_config"

    ROLE_PERMISSIONS = {
        Role.OPERATOR: [Permission.VIEW, Permission.MANUAL_OVERRIDE],
        Role.ENGINEER: [Permission.VIEW, Permission.MANUAL_OVERRIDE, Permission.ADJUST_PARAMS],
        Role.ADMIN: [Permission.VIEW, Permission.MANUAL_OVERRIDE, Permission.ADJUST_PARAMS, Permission.FULL_CONFIG],
    }

    def __init__(self, config: dict):
        self.enabled = config.get("enable", False)
        self.secret_key = config.get("secret_key", "defect-detection-secret")
        self.token_expire_hours = config.get("token_expire_hours", 8)
        self.users: Dict[str, Dict] = {}
        users_config = config.get("users", [])
        if not users_config:
            users_config = [
                {"username": "operator", "password": "operator123", "role": "operator"},
                {"username": "engineer", "password": "engineer123", "role": "engineer"},
                {"username": "admin", "password": "admin123", "role": "admin"},
            ]
        for user in users_config:
            role = self.Role(user["role"])
            self.users[user["username"]] = {
                "password": user["password"],
                "role": role,
                "permissions": self.ROLE_PERMISSIONS[role],
            }

    def _generate_token(self, username: str, role: Role) -> str:
        expiry = int(time.time()) + self.token_expire_hours * 3600
        payload = {
            "username": username,
            "role": role.value,
            "expiry": expiry,
        }
        payload_json = json.dumps(payload, separators=(",", ":"))
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload_b64}.{signature}"

    def authenticate(self, username: str, password: str) -> Optional[str]:
        user = self.users.get(username)
        if user is None or user["password"] != password:
            logger.warning(f"Authentication failed for user: {username}")
            return None
        token = self._generate_token(username, user["role"])
        logger.info(f"User authenticated: {username}")
        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            payload_b64, signature = parts
            expected_signature = hmac.new(
                self.secret_key.encode("utf-8"),
                payload_b64.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Token signature verification failed")
                return None
            payload_json = base64.b64decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)
            if payload.get("expiry", 0) < int(time.time()):
                logger.warning("Token expired")
                return None
            role = self.Role(payload["role"])
            permissions = self.ROLE_PERMISSIONS[role]
            return {
                "username": payload["username"],
                "role": role.value,
                "permissions": [p.value for p in permissions],
            }
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def check_permission(self, token: str, permission) -> bool:
        info = self.verify_token(token)
        if info is None:
            return False
        perm_value = permission.value if isinstance(permission, self.Permission) else permission
        return perm_value in info["permissions"]

    def get_user_info(self, token: str) -> Optional[Dict]:
        return self.verify_token(token)

    def list_users(self) -> List[Dict]:
        return [
            {
                "username": username,
                "role": user["role"].value,
                "permissions": [p.value for p in user["permissions"]],
            }
            for username, user in self.users.items()
        ]

    def add_user(self, username: str, password: str, role_str: str) -> bool:
        if username in self.users:
            logger.warning(f"User already exists: {username}")
            return False
        role = self.Role(role_str)
        self.users[username] = {
            "password": password,
            "role": role,
            "permissions": self.ROLE_PERMISSIONS[role],
        }
        logger.info(f"User added: {username}")
        return True

    def remove_user(self, username: str) -> bool:
        if username not in self.users:
            logger.warning(f"User not found: {username}")
            return False
        del self.users[username]
        logger.info(f"User removed: {username}")
        return True
