"""Router xác thực."""

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_auth_service
from ..schemas import auth as auth_schema
from ..schemas.common import Message
from ..services.auth_service import AuthService

router = APIRouter()


@router.post("/register", response_model=auth_schema.AuthResponse, status_code=201)
def register(payload: auth_schema.RegisterRequest, auth_service: Annotated[AuthService, Depends(get_auth_service)]):
    try:
        user = auth_service.register(payload.username, payload.email, payload.password)
        tokens = auth_service.issue_tokens(user)
        return auth_schema.AuthResponse(user_id=user.id, token=auth_schema.TokenPair(**tokens))
    except Exception as e:
        # Log error for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Registration error: {e}", exc_info=True)
        raise


@router.post("/login", response_model=auth_schema.AuthResponse)
def login(payload: auth_schema.LoginRequest, auth_service: Annotated[AuthService, Depends(get_auth_service)]):
    user = auth_service.authenticate(payload.username_or_email, payload.password)
    tokens = auth_service.issue_tokens(user)
    return auth_schema.AuthResponse(user_id=user.id, token=auth_schema.TokenPair(**tokens))


@router.post("/refresh", response_model=auth_schema.TokenPair)
def refresh(payload: auth_schema.RefreshRequest, auth_service: Annotated[AuthService, Depends(get_auth_service)]):
    tokens = auth_service.refresh(payload.refresh_token)
    return auth_schema.TokenPair(**tokens)


@router.post("/logout", response_model=Message)
def logout(payload: auth_schema.RefreshRequest, auth_service: Annotated[AuthService, Depends(get_auth_service)]):
    auth_service.revoke_refresh_token(payload.refresh_token)
    return Message(message="Logged out")

