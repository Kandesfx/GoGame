"""Kết nối cơ sở dữ liệu PostgreSQL & MongoDB."""

import asyncio
import logging
from collections.abc import Generator
from typing import AsyncIterator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import get_settings

# Cấu hình logging sớm để giảm verbosity của SQLAlchemy
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)

settings = get_settings()

logger = logging.getLogger(__name__)

# SQLAlchemy setup với connection timeout và better error handling
# connect_args: thêm connect_timeout để tránh hang quá lâu
# pool_pre_ping: test connection trước khi dùng
# pool_recycle: recycle connections sau 1 giờ để tránh stale connections
try:
    engine = create_engine(
        settings.postgres_dsn,
        echo=False,
        future=True,
        pool_pre_ping=True,  # Test connection trước khi dùng
        pool_recycle=3600,  # Recycle connections sau 1 giờ
        connect_args={
            "connect_timeout": 5,  # 5 seconds timeout khi connect
        },
    )
    logger.info(f"✅ [POSTGRES] Engine created with DSN: {settings.postgres_dsn.split('@')[0]}@***")
except Exception as e:
    logger.error(f"❌ [POSTGRES] Failed to create engine: {e}")
    raise

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=Session)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency Sync DB session cho FastAPI.
    
    Raises HTTPException với message rõ ràng nếu không thể kết nối database.
    """
    from fastapi import HTTPException, status
    
    session = SessionLocal()
    try:
        # Test connection bằng cách execute một query đơn giản
        # Nếu fail, sẽ raise exception với message rõ ràng
        session.execute(text("SELECT 1"))
        yield session
    except Exception as e:
        session.rollback()
        error_msg = str(e)
        
        # Kiểm tra xem có phải là authentication error không (từ dependencies)
        # Authentication errors không phải là database errors
        if any(keyword in error_msg for keyword in ["401", "Invalid token", "Token revoked", "Missing credentials", "Invalid token payload", "User not found"]):
            # Đây là authentication error, không phải database error
            # Re-raise để FastAPI xử lý đúng (401 Unauthorized)
            # Không log như database error
            raise
        
        # Parse error để đưa ra message hữu ích hơn cho database errors
        if "password authentication failed" in error_msg.lower():
            logger.error(
                "❌ [POSTGRES] Password authentication failed. "
                "Please check your DATABASE_URL or POSTGRES_DSN environment variable. "
                f"Current DSN: {settings.postgres_dsn.split('@')[0]}@***"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Database connection failed: Password authentication failed. "
                    "Please check your database credentials in DATABASE_URL or POSTGRES_DSN environment variable. "
                    "See backend/env.example for configuration example."
                )
            ) from e
        elif "connection" in error_msg.lower() and "failed" in error_msg.lower():
            logger.error(
                f"❌ [POSTGRES] Connection failed: {error_msg}. "
                f"DSN: {settings.postgres_dsn.split('@')[0]}@***"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Database connection failed: {error_msg}. "
                    "Please ensure PostgreSQL is running and accessible. "
                    "Check your DATABASE_URL or POSTGRES_DSN environment variable."
                )
            ) from e
        else:
            # Chỉ log như database error nếu thực sự là database error
            # Kiểm tra xem có phải là SQLAlchemy/psycopg error không
            error_type = type(e).__name__
            if any(db_error_type in error_type for db_error_type in ["OperationalError", "DatabaseError", "IntegrityError", "ProgrammingError"]):
                logger.error(f"❌ [POSTGRES] Database error: {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Database error: {error_msg}"
                ) from e
            else:
                # Không phải database error, re-raise để xử lý ở layer khác
                raise
    finally:
        session.close()


# Mongo setup
_mongo_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    """Lazy khởi tạo Mongo client để tránh tạo nhiều kết nối."""
    import time
    logger = logging.getLogger(__name__)
    
    global _mongo_client  # noqa: PLW0603
    if _mongo_client is None:
        start_time = time.time()
        try:
            # Tạo client với serverSelectionTimeoutMS ngắn để tránh block quá lâu
            # connectTimeoutMS cũng ngắn để không block khi MongoDB không chạy
            _mongo_client = AsyncIOMotorClient(
                settings.mongo_dsn,
                serverSelectionTimeoutMS=2000,  # 2 seconds timeout - ngắn hơn
                connectTimeoutMS=2000,  # 2 seconds connection timeout
                socketTimeoutMS=2000,  # 2 seconds socket timeout
            )
            elapsed = time.time() - start_time
            if elapsed > 0.1:
                logger.warning(f"⏱️ [MONGO] Creating Mongo client took {elapsed:.3f}s")
            else:
                logger.debug(f"✅ [MONGO] Mongo client created in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"⚠️ [MONGO] Failed to create Mongo client after {elapsed:.3f}s: {e}")
            # KHÔNG raise - cho phép app chạy mà không có MongoDB
            # Tạo một dummy client với timeout rất ngắn để fail nhanh nếu MongoDB không chạy
            try:
                _mongo_client = AsyncIOMotorClient(
                    settings.mongo_dsn,
                    serverSelectionTimeoutMS=1,  # Rất ngắn để fail nhanh
                    connectTimeoutMS=1,
                )
            except Exception:
                # Nếu vẫn fail, tạo client với default settings (sẽ fail khi dùng)
                _mongo_client = AsyncIOMotorClient(settings.mongo_dsn)
    return _mongo_client


async def get_mongo_db() -> AsyncIterator[AsyncIOMotorDatabase | None]:
    """Dependency trả về database MongoDB.
    
    KHÔNG block nếu MongoDB không available - chỉ log warning và yield None.
    KHÔNG raise exception - luôn yield (có thể là None) để tránh generator error.
    """
    import time
    logger = logging.getLogger(__name__)
    
    logger.info("🟡 [MONGO] get_mongo_db called - getting MongoDB connection...")
    start_time = time.time()
    
    db = None
    try:
        client = get_mongo_client()
        if client:
            db = client[settings.mongo_database]
        
        # KHÔNG test connection - chỉ lấy client và yield ngay để không block
        # MongoDB sẽ được test khi thực sự cần dùng
        
        elapsed = time.time() - start_time
        if elapsed > 0.05:
            logger.warning(f"⏱️ [MONGO] get_mongo_db took {elapsed:.3f}s to get client")
        else:
            logger.info(f"✅ [MONGO] get_mongo_db got client in {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"⚠️ [MONGO] get_mongo_db failed after {elapsed:.3f}s: {e} - continuing without MongoDB")
        # KHÔNG raise - set db = None để app vẫn chạy được
        # MatchService sẽ handle None mongo_db
        db = None
    
    try:
        yield db
    except GeneratorExit:
        # Generator đang được đóng - không làm gì
        raise
    except Exception as e:
        # Nếu có exception trong quá trình sử dụng, log nhưng không raise
        logger.error(f"❌ [MONGO] Exception during get_mongo_db usage: {e}", exc_info=True)
        # KHÔNG raise - để tránh generator error
        raise
    finally:
        # không đóng kết nối để tái sử dụng (Motor handle connection pool)
        pass
