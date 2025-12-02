"""Cấu hình backend sử dụng Pydantic BaseSettings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import AnyHttpUrl, EmailStr, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Định nghĩa toàn bộ biến môi trường cần thiết."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "GoGame Backend"
    debug: bool = False
    allowed_hosts: List[AnyHttpUrl] = Field(default_factory=list)
    cors_origins: List[AnyHttpUrl] = Field(default_factory=list)

    # Database
    postgres_dsn: str = "postgresql+psycopg://postgres:postgres@localhost:5432/gogame"
    mongo_dsn: str = "mongodb://localhost:27017"
    mongo_database: str = "gogame"
    
    @field_validator('postgres_dsn', mode='before')
    @classmethod
    def convert_postgres_dsn(cls, v: str) -> str:
        """Convert postgresql:// to postgresql+psycopg:// if needed."""
        if isinstance(v, str) and v.startswith("postgresql://") and not v.startswith("postgresql+psycopg://"):
            return v.replace("postgresql://", "postgresql+psycopg://", 1)
        return v

    # JWT
    jwt_secret_key: str = "change-me"
    jwt_refresh_secret_key: str = "change-me-refresh"
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = 480  # Access token hết hạn sau 8 giờ (đủ cho game session dài, tránh gián đoạn)
    refresh_token_exp_days: int = 7  # Refresh token hết hạn sau 7 ngày (sliding session)
    session_inactivity_timeout_days: int = 7  # Đánh out nếu không có request trong 7 ngày

    # Email (tương lai)
    support_email: Optional[EmailStr] = None

    # S3 / MinIO
    s3_endpoint: Optional[AnyHttpUrl] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket_models: str = "gogame-models"

    # AI engine integration
    gogame_py_module: str = "gogame_py"
    ai_move_timeout_seconds: float = 30.0  # Timeout cho AI move selection
    ai_move_retry_count: int = 1  # Số lần retry nếu AI move fail

    # Evaluation cache settings
    eval_cache_max_size: int = 2000  # Số lượng entries tối đa trong cache
    eval_cache_ttl_seconds: float = 3600.0  # Time-to-live cho cache entries (1 hour)


@lru_cache
def get_settings() -> Settings:
    """Lấy config đã cache (theo khuyến nghị FastAPI)."""

    return Settings()
