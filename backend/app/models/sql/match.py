"""Bảng matches."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    black_player_id: Mapped[Optional[str]] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.id", ondelete="SET NULL"))
    white_player_id: Mapped[Optional[str]] = mapped_column(PG_UUID(as_uuid=False), ForeignKey("users.id", ondelete="SET NULL"))
    ai_level: Mapped[Optional[int]] = mapped_column(Integer)
    board_size: Mapped[int] = mapped_column(Integer, default=9)
    result: Mapped[Optional[str]] = mapped_column(String(32))
    room_code: Mapped[Optional[str]] = mapped_column(String(6), unique=True, index=True)  # Mã bàn 6 ký tự
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    sgf_id: Mapped[Optional[str]] = mapped_column(String(64))
    premium_analysis_id: Mapped[Optional[str]] = mapped_column(String(64))
    # Time control cho PvP matches
    time_control_minutes: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # Thời gian tổng cho mỗi người chơi (phút)
    black_time_remaining_seconds: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # Thời gian còn lại của Black (giây)
    white_time_remaining_seconds: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # Thời gian còn lại của White (giây)
    last_move_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)  # Thời điểm nước đi cuối cùng
    # ELO changes (chỉ cho PvP matches)
    black_elo_change: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # ELO change của Black player
    white_elo_change: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # ELO change của White player
    # Ready status cho matchmaking (chỉ cho PvP matches)
    black_ready: Mapped[bool] = mapped_column(default=False)  # Black player đã sẵn sàng
    white_ready: Mapped[bool] = mapped_column(default=False)  # White player đã sẵn sàng

    black_player = relationship("User", foreign_keys=[black_player_id], back_populates="matches_as_black")
    white_player = relationship("User", foreign_keys=[white_player_id], back_populates="matches_as_white")

