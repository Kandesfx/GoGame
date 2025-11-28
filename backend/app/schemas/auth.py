"""Schema cho module xác thực."""

from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, constr


class RegisterRequest(BaseModel):
    username: constr(min_length=3, max_length=32)
    email: EmailStr
    password: constr(min_length=8)


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., examples=["player1", "player@example.com"])
    password: constr(min_length=8)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class AuthResponse(BaseModel):
    user_id: UUID
    token: TokenPair

