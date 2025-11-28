"""Schema người dùng."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, constr


class UserBase(BaseModel):
    id: UUID
    username: constr(min_length=3, max_length=32)
    email: EmailStr
    elo_rating: int = 1500
    coins: int = 0
    display_name: str | None = None
    avatar_url: str | None = None
    preferences: dict | None = None
    created_at: datetime
    last_login: datetime | None = None


class UserPublic(BaseModel):
    id: UUID
    username: str
    display_name: str | None = None
    elo_rating: int


class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=64)
    avatar_url: str | None = None
    preferences: dict | None = None

