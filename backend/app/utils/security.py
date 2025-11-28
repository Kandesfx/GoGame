"""Bảo mật: hash mật khẩu và JWT helpers."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from ..config import Settings

password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return password_hasher.verify(hashed, password)
    except VerifyMismatchError:
        return False


def _expiry(now: datetime, delta_seconds: int) -> datetime:
    return now + timedelta(seconds=delta_seconds)


def create_access_token(subject: str, settings: Settings, additional_claims: Optional[Dict[str, Any]] = None) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(_expiry(now, settings.access_token_exp_minutes * 60).timestamp()),
        **(additional_claims or {}),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str, settings: Settings, token_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": subject,
        "jti": token_id,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(_expiry(now, settings.refresh_token_exp_days * 24 * 60 * 60).timestamp()),
        **(additional_claims or {}),
    }
    return jwt.encode(payload, settings.jwt_refresh_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> Dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def decode_refresh_token(token: str, settings: Settings) -> Dict[str, Any]:
    return jwt.decode(token, settings.jwt_refresh_secret_key, algorithms=[settings.jwt_algorithm])

