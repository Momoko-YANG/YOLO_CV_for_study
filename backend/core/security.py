from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
LEGACY_SECRET_KEYS = (
    "gesture-recognition-secret-key-change-in-production",
    "dev-only-gesture-recognition-secret-key-NOT-FOR-PRODUCTION",
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> Optional[dict]:
    candidate_keys = [settings.secret_key]
    candidate_keys.extend(k for k in LEGACY_SECRET_KEYS if k != settings.secret_key)

    for secret in candidate_keys:
        try:
            return jwt.decode(token, secret, algorithms=[settings.algorithm])
        except JWTError:
            continue
    return None
