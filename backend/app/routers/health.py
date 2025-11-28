"""Endpoints kiểm tra trạng thái hệ thống."""

from fastapi import APIRouter, Depends

from ..database import get_db, get_mongo_db

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
async def health_check(db=Depends(get_db), mongo=Depends(get_mongo_db)) -> dict[str, str]:
    del db  # kiểm tra khởi tạo session thành công
    del mongo  # đảm bảo dependency chạy được
    return {"status": "ok"}

