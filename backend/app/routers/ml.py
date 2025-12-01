"""Router quản trị ML."""

from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends

from ..dependencies import get_current_user, get_ml_service
from ..models.sql import user as user_models
from ..schemas import ml as ml_schema
from ..services.ml_service import MLService

router = APIRouter()


def ensure_admin(current_user) -> None:
    # TODO: kiểm tra quyền admin (tạm thời cho phép tất cả)
    return None


@router.post("/train", response_model=dict)
async def trigger_training(
    payload: ml_schema.TrainRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[MLService, Depends(get_ml_service)],
):
    ensure_admin(current_user)
    return await service.trigger_training(payload)


@router.get("/models", response_model=List[ml_schema.ModelVersion])
async def list_models(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[MLService, Depends(get_ml_service)],
):
    ensure_admin(current_user)
    return await service.list_models()


@router.post("/models/{model_id}/promote", response_model=dict)
async def promote_model(
    model_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[MLService, Depends(get_ml_service)],
):
    ensure_admin(current_user)
    return await service.promote_model(model_id)

