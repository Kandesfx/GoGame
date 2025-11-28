"""Schemas cho matchmaking."""

from pydantic import BaseModel, Field


class JoinQueueRequest(BaseModel):
    board_size: int = Field(..., ge=9, le=19, description="Kích thước bàn cờ (9, 13, 19)")


class QueueStatusResponse(BaseModel):
    in_queue: bool
    board_size: int | None = None
    elo_rating: int | None = None
    wait_time: int | None = None
    queue_size: int | None = None
    elo_range: int | None = None


class QueueStatsResponse(BaseModel):
    total_players: int
    by_board_size: dict[int, int]
    running: bool

