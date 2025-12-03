"""Entry point FastAPI cho GoGame backend."""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import auth, coins, health, matches, matchmaking, ml, premium, statistics, users
from .tasks import background
from .dependencies import get_db, get_matchmaking_service

# Configure logging: giảm verbosity của SQLAlchemy và các thư viện khác
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Giảm verbosity của SQLAlchemy (chỉ hiển thị WARNING trở lên)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)

# BẬT uvicorn access logs để debug (tạm thời)
logging.getLogger('uvicorn.access').setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Khởi tạo FastAPI application."""

    settings = get_settings()

    app = FastAPI(title=settings.app_name, debug=settings.debug)

    # Middleware để log tất cả requests - THÊM TRƯỚC CORS
    # LƯU Ý: Trong FastAPI, middleware được thêm theo thứ tự ngược lại
    # Middleware này được thêm TRƯỚC CORS nên sẽ chạy SAU CORS (last added = first executed)
    # Nhưng vẫn log được request vì nó wrap toàn bộ request flow
    @app.middleware("http")
    async def log_requests(request, call_next):
        """Log tất cả requests để debug."""
        import time
        start_time = time.time()
        
        # Log request với đầy đủ thông tin
        origin = request.headers.get("origin", "no-origin")
        user_agent = request.headers.get("user-agent", "no-ua")
        logger.info(f"📥 [REQUEST] {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'} | Origin: {origin} | UA: {user_agent[:50]}")
        
        try:
            response = await call_next(request)
            elapsed = time.time() - start_time
            logger.info(f"📤 [RESPONSE] {request.method} {request.url.path} -> {response.status_code} in {elapsed:.3f}s")
            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ [ERROR] {request.method} {request.url.path} failed after {elapsed:.3f}s: {e}", exc_info=True)
            raise

    # CORS: Always enable for development, or use configured origins
    cors_origins = [str(origin).rstrip('/') for origin in settings.cors_origins_list] if settings.cors_origins_list else ["*"]
    
    # Log CORS settings
    allowed_origins = cors_origins if cors_origins != ["*"] else ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"]
    logger.info(f"🌐 CORS allowed origins: {allowed_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handler để đảm bảo CORS headers được thêm vào error responses
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler để đảm bảo CORS headers trong error responses."""
        import traceback
        logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
        
        # Tạo response với CORS headers
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
        
        # Thêm CORS headers manually
        origin = request.headers.get("origin")
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        """HTTP exception handler với CORS headers."""
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
        
        # Thêm CORS headers
        origin = request.headers.get("origin")
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        """Validation exception handler với CORS headers."""
        response = JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )
        
        # Thêm CORS headers
        origin = request.headers.get("origin")
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

    # Đăng ký routers
    app.include_router(health.router)
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(users.router, prefix="/users", tags=["users"])
    app.include_router(matches.router, prefix="/matches", tags=["matches"])
    app.include_router(coins.router, prefix="/coins", tags=["coins"])
    app.include_router(premium.router, prefix="/premium", tags=["premium"])
    app.include_router(ml.router, prefix="/ml", tags=["ml"])
    app.include_router(statistics.router, prefix="/statistics", tags=["statistics"])
    app.include_router(matchmaking.router, prefix="/matchmaking", tags=["matchmaking"])

    # Background tasks lifecycle
    @app.on_event("startup")
    async def startup_background_tasks():
        """Start background tasks khi app khởi động."""
        logger.info("Starting background tasks...")
        
        # Start cache cleanup task
        asyncio.create_task(background.cleanup_evaluation_cache(interval_seconds=300))
        
        # Start statistics update task
        asyncio.create_task(background.update_user_statistics(interval_seconds=3600))
        
        # Start matchmaking service
        # Note: MatchmakingService sẽ tự start khi có user join queue
        # Không cần start ở đây vì service được tạo mới mỗi request
        logger.info("Matchmaking service will start automatically when users join queue")
        
        logger.info("Background tasks started")

    @app.on_event("shutdown")
    async def shutdown_background_tasks():
        """Cleanup khi app shutdown."""
        logger.info("Shutting down background tasks...")
        
        # Stop matchmaking service
        # Note: Sử dụng shared state nên không cần db session
        try:
            from .services.matchmaking_service import MatchmakingService
            # Tạo một instance tạm để gọi stop (không cần db vì chỉ dùng shared state)
            from .database import SessionLocal
            temp_db = SessionLocal()
            try:
                matchmaking_service = MatchmakingService(temp_db)
                matchmaking_service.stop_matching_task()
                logger.info("Matchmaking service stopped")
            finally:
                temp_db.close()
        except Exception as e:
            logger.error(f"Failed to stop matchmaking service: {e}", exc_info=True)

    return app


app = create_app()

