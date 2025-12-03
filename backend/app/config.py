"""Cấu hình backend sử dụng Pydantic BaseSettings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AnyHttpUrl, EmailStr, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_file_path() -> str:
    """Tìm file .env trong thư mục backend hoặc thư mục hiện tại."""
    # Lấy thư mục chứa file config.py (backend/app/)
    config_dir = Path(__file__).parent.parent
    
    # Tìm .env trong thư mục backend
    env_file = config_dir / ".env"
    if env_file.exists():
        return str(env_file)
    
    # Fallback: tìm trong thư mục hiện tại
    current_dir_env = Path(".env")
    if current_dir_env.exists():
        return str(current_dir_env)
    
    # Nếu không tìm thấy, trả về đường dẫn mặc định trong backend
    return str(config_dir / ".env")


class Settings(BaseSettings):
    """Định nghĩa toàn bộ biến môi trường cần thiết."""

    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "GoGame Backend"
    debug: bool = False
    allowed_hosts: List[AnyHttpUrl] = Field(default_factory=list)
    
    @field_validator('debug', mode='before')
    @classmethod
    def parse_debug(cls, v) -> bool:
        """Parse debug from various string formats.
        
        Handles:
        - None/not provided: returns False (default)
        - Boolean: returns as-is
        - String: parses common boolean representations
        - Other types: returns False
        """
        # If not provided or None, use default (False)
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ('true', '1', 'yes', 'on'):
                return True
            if v_lower in ('false', '0', 'no', 'off', 'warn', ''):
                return False
        # Default to False for any other value
        return False
    # Store as str to avoid JSON parsing, then parse in model_post_init
    cors_origins: Optional[str] = Field(default=None, validation_alias="CORS_ORIGINS")
    cors_origins_parsed: List[AnyHttpUrl] = Field(default_factory=list, exclude=True)
    
    def model_post_init(self, __context) -> None:
        """Parse environment variables for database and CORS.
        
        Handles:
        - DATABASE_URL (production/Fly.io) - priority 1
        - POSTGRES_DSN (local development from .env) - priority 2
        - CORS_ORIGINS parsing (JSON array or comma-separated string)
        """
        import os
        
        # Load .env file explicitly to ensure it's read (for local development)
        try:
            from dotenv import load_dotenv
            # Load from backend/.env
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)  # Don't override existing env vars
        except ImportError:
            pass  # python-dotenv not installed, rely on Pydantic's built-in loading
        
        # Database connection: Read DATABASE_URL first (production), then POSTGRES_DSN (local)
        default_local_dsn = "postgresql+psycopg://postgres:postgres@localhost:5432/gogame"
        
        # Only update if we still have the default value
        if self.postgres_dsn == default_local_dsn:
            # Priority 1: DATABASE_URL (production/Fly.io)
            database_url = os.getenv("DATABASE_URL")
            if database_url and database_url.strip():
                # Apply the same conversion as the validator
                if database_url.startswith("postgresql://") and not database_url.startswith("postgresql+psycopg://"):
                    database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
                self.postgres_dsn = database_url
            else:
                # Priority 2: POSTGRES_DSN (local development from .env)
                postgres_dsn = os.getenv("POSTGRES_DSN")
                if postgres_dsn and postgres_dsn.strip():
                    # Apply the same conversion as the validator
                    if postgres_dsn.startswith("postgresql://") and not postgres_dsn.startswith("postgresql+psycopg://"):
                        postgres_dsn = postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
                    self.postgres_dsn = postgres_dsn
        
        # Parse CORS_ORIGINS - supports both JSON array and comma-separated string
        cors_raw = self.cors_origins or os.getenv("CORS_ORIGINS")
        if cors_raw and isinstance(cors_raw, str) and cors_raw.strip():
            # Try JSON first (for production environments like Fly.io)
            try:
                import json
                parsed = json.loads(cors_raw)
                if isinstance(parsed, list):
                    # Strip trailing slashes from origins and validate
                    valid_origins = []
                    for origin in parsed:
                        if isinstance(origin, str) and origin.strip():
                            try:
                                valid_origins.append(AnyHttpUrl(origin.rstrip('/')))
                            except Exception:
                                # Skip invalid URLs
                                continue
                    self.cors_origins_parsed = valid_origins
                    return
            except (json.JSONDecodeError, ValueError, Exception):
                # Not JSON, continue to comma-separated parsing
                pass
            
            # If not JSON, treat as comma-separated string (for local development)
            origins = []
            for origin in cors_raw.split(','):
                origin = origin.strip().rstrip('/')
                if origin:
                    try:
                        origins.append(AnyHttpUrl(origin))
                    except Exception:
                        # Skip invalid URLs
                        continue
            self.cors_origins_parsed = origins
    
    @property
    def cors_origins_list(self) -> List[AnyHttpUrl]:
        """Get parsed CORS origins as list."""
        return self.cors_origins_parsed

    # Database
    # Read from DATABASE_URL (Fly.io standard) or POSTGRES_DSN
    # Sử dụng Field với validation_alias để đọc cả DATABASE_URL và POSTGRES_DSN
    postgres_dsn: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/gogame"
    )
    mongo_dsn: str = "mongodb://localhost:27017"
    mongo_database: str = "gogame"
    
    @field_validator('postgres_dsn', mode='before')
    @classmethod
    def convert_postgres_dsn(cls, v) -> str:
        """Convert postgresql:// to postgresql+psycopg:// if needed.
        
        Handles:
        - None values (returns default)
        - Empty strings (returns default)
        - postgresql:// URLs (converts to postgresql+psycopg://)
        - Already correct URLs (returns as-is)
        """
        if v is None:
            return "postgresql+psycopg://postgres:postgres@localhost:5432/gogame"
        if not isinstance(v, str):
            return str(v)
        if not v.strip():
            return "postgresql+psycopg://postgres:postgres@localhost:5432/gogame"
        # Convert postgresql:// to postgresql+psycopg:// if needed
        if v.startswith("postgresql://") and not v.startswith("postgresql+psycopg://"):
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
    """Lấy config đã cache (theo khuyến nghị FastAPI).
    
    Note: Cache sẽ được clear khi code reload (uvicorn --reload).
    Nếu thay đổi .env file, cần restart server hoặc clear cache:
    from app.config import get_settings
    get_settings.cache_clear()
    """
    return Settings()
