"""Service người dùng."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from ..models.sql import user as user_model


class UserService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_user(self, user_id: UUID) -> Optional[user_model.User]:
        return self.db.get(user_model.User, str(user_id))

    def update_user(self, user_id: UUID, *, display_name: str | None, avatar_url: str | None, preferences: dict | None):
        user = self.get_user(user_id)
        if not user:
            return None

        if display_name is not None:
            user.display_name = display_name
        if avatar_url is not None:
            user.avatar_url = avatar_url
        if preferences is not None:
            user.preferences = preferences

        self.db.commit()
        self.db.refresh(user)
        return user

