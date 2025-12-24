"""Các dependency dùng chung cho FastAPI routers."""

from typing import Annotated, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_db, get_mongo_db
from .models.sql import user as user_models
from .services.auth_service import AuthService
from .services.coin_service import CoinService
from .services.match_service import MatchService
from .services.premium_service import PremiumService
from .services.ml_service import MLService
from .services.user_service import UserService
from .services.statistics_service import StatisticsService
from .services.matchmaking_service import MatchmakingService
from .services.payment_service import PaymentService

# Import ML position analysis service safely (optional dependency)
try:
    from .services.ml_position_analysis_service import get_ml_position_analysis_service
except ImportError:
    # ML dependencies not available - service will be None
    get_ml_position_analysis_service = None

BearerToken = HTTPBearer(auto_error=False)


def get_auth_service(db: Annotated[Session, Depends(get_db)]) -> AuthService:
    """Khởi tạo AuthService với session hiện tại."""

    return AuthService(db=db, settings=get_settings())


def get_user_service(db: Annotated[Session, Depends(get_db)]) -> UserService:
    return UserService(db=db)


def get_coin_service(db: Annotated[Session, Depends(get_db)]) -> CoinService:
    return CoinService(db)


def get_match_service(
    db: Annotated[Session, Depends(get_db)],
    mongo: Annotated[AsyncIOMotorDatabase, Depends(get_mongo_db)],
) -> MatchService:
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔵 [DEPS] get_match_service called - resolving MongoDB dependency...")
    start_time = time.time()
    
    try:
        # mongo có thể là None nếu MongoDB không available - MatchService sẽ handle
        service = MatchService(db=db, mongo_db=mongo, settings=get_settings())
        elapsed = time.time() - start_time
        if elapsed > 0.05:
            logger.warning(f"⏱️ [DEPS] get_match_service took {elapsed:.3f}s")
        else:
            logger.info(f"✅ [DEPS] get_match_service completed in {elapsed:.3f}s")
        return service
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ [DEPS] get_match_service failed after {elapsed:.3f}s: {e}", exc_info=True)
        raise


def get_premium_service(
    db: Annotated[Session, Depends(get_db)],
    mongo: Annotated[AsyncIOMotorDatabase, Depends(get_mongo_db)],
) -> PremiumService:
    return PremiumService(db=db, mongo_db=mongo, settings=get_settings())


def get_ml_service(
    db: Annotated[Session, Depends(get_db)],
    mongo: Annotated[AsyncIOMotorDatabase, Depends(get_mongo_db)],
) -> MLService:
    return MLService(db=db, mongo_db=mongo, settings=get_settings())


def get_statistics_service(db: Annotated[Session, Depends(get_db)]) -> StatisticsService:
    return StatisticsService(db=db)


def get_matchmaking_service(db: Annotated[Session, Depends(get_db)]) -> MatchmakingService:
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🟢 [DEPS] get_matchmaking_service called...")
    start_time = time.time()
    
    try:
        service = MatchmakingService(db=db)
        elapsed = time.time() - start_time
        if elapsed > 0.05:
            logger.warning(f"⏱️ [DEPS] get_matchmaking_service took {elapsed:.3f}s")
        else:
            logger.info(f"✅ [DEPS] get_matchmaking_service completed in {elapsed:.3f}s")
        return service
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ [DEPS] get_matchmaking_service failed after {elapsed:.3f}s: {e}", exc_info=True)
        raise


def get_payment_service(db: Annotated[Session, Depends(get_db)]) -> PaymentService:
    return PaymentService(db=db)


def get_ml_position_analysis_service_dep():
    """Dependency để get ML position analysis service."""
    if get_ml_position_analysis_service is None:
        return None
    return get_ml_position_analysis_service()


def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Security(BearerToken)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> user_models.User:
    """Giải mã JWT và trả về người dùng hiện tại.
    
    QUAN TRỌNG: Tự động extend session cho MỌI request từ user đang active.
    Backend hoàn toàn kiểm soát session - chỉ đánh out khi không có request trong thời gian dài.
    """
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")

    token = credentials.credentials
    decode_start = time.time()
    payload = auth_service.decode_access_token(token)
    decode_time = time.time() - decode_start
    if decode_time > 0.01:
        logger.warning(f"⏱️ [AUTH] decode_access_token took {decode_time:.3f}s")
    
    try:
        user_id = UUID(payload.get("sub", ""))
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload") from None
    
    get_user_start = time.time()
    user = auth_service.get_user(user_id)
    get_user_time = time.time() - get_user_start
    if get_user_time > 0.01:
        logger.warning(f"⏱️ [AUTH] get_user took {get_user_time:.3f}s")
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    
    # QUAN TRỌNG: Extend session cho MỌI request (không có điều kiện)
    # Backend hoàn toàn kiểm soát - session chỉ hết hạn khi không có request trong thời gian dài
    extend_start = time.time()
    try:
        auth_service.extend_session_if_active(user_id)
        extend_time = time.time() - extend_start
        if extend_time > 0.1:
            logger.warning(f"⏱️ [AUTH] extend_session_if_active took {extend_time:.3f}s")
    except Exception as e:
        # Log lỗi nhưng không fail request (session extension là optional)
        # Nếu extend fail, user vẫn có thể tiếp tục dùng access token hiện tại
        extend_time = time.time() - extend_start
        logger.warning(f"❌ [AUTH] Failed to extend session for user {user_id} after {extend_time:.3f}s: {e}")
    
    total_time = time.time() - start_time
    if total_time > 0.1:
        logger.warning(f"⏱️ [AUTH] get_current_user total took {total_time:.3f}s")
    
    return user

