"""Bảng người dùng."""

from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    elo_rating: Mapped[int] = mapped_column(Integer, default=1500)
    coins: Mapped[int] = mapped_column(Integer, default=0)
    display_name: Mapped[str | None] = mapped_column(String(64))
    avatar_url: Mapped[str | None] = mapped_column(String(255))
    preferences: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    matches_as_black: Mapped[List["Match"]] = relationship(
        back_populates="black_player",
        cascade="all,delete",
        foreign_keys="Match.black_player_id",
    )
    matches_as_white: Mapped[List["Match"]] = relationship(
        back_populates="white_player",
        cascade="all,delete",
        foreign_keys="Match.white_player_id",
    )
    premium_subscription: Mapped["PremiumSubscription | None"] = relationship(
        "PremiumSubscription", back_populates="user", uselist=False, cascade="all,delete"
    )

