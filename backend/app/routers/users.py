"""Router người dùng."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user, get_user_service
from ..models.sql import user as user_models
from ..schemas import users as user_schema
from ..services.user_service import UserService

router = APIRouter()


def _to_user_schema(user) -> user_schema.UserBase:
    return user_schema.UserBase(
        id=UUID(user.id),
        username=user.username,
        email=user.email,
        elo_rating=user.elo_rating,
        coins=user.coins,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        preferences=user.preferences,
        created_at=user.created_at,
        last_login=user.last_login,
    )


@router.get("/me", response_model=user_schema.UserBase)
def get_me(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
):
    return _to_user_schema(current_user)


@router.patch("/me", response_model=user_schema.UserBase)
def update_me(
    payload: user_schema.UserUpdateRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    user_service: Annotated[UserService, Depends(get_user_service)],
):
    updated = user_service.update_user(
        UUID(current_user.id),
        display_name=payload.display_name,
        avatar_url=payload.avatar_url,
        preferences=payload.preferences,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return _to_user_schema(updated)


@router.get(
    "/{user_id}",
    response_model=user_schema.UserPublic,
    summary="Lấy thông tin công khai người dùng",
)
def get_public_profile(
    user_id: UUID,
    user_service: Annotated[UserService, Depends(get_user_service)],
):
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user_schema.UserPublic(
        id=user_id,
        username=user.username,
        display_name=user.display_name,
        elo_rating=user.elo_rating,
    )

