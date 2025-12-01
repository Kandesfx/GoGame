"""Schema cho trận đấu."""

from datetime import datetime
from typing import Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field, conint


class MatchBase(BaseModel):
    id: UUID
    board_size: conint(ge=9, le=19) = 9
    ai_level: int | None = None
    result: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class MatchCreateAIRequest(BaseModel):
    level: conint(ge=1, le=4)
    board_size: conint(ge=9, le=19) = 9
    player_color: Literal["black", "white"] = "black"


class MatchCreatePVPRequest(BaseModel):
    board_size: conint(ge=9, le=19) = 9
    time_control_minutes: conint(ge=1, le=60) = Field(default=10, description="Thời gian tổng cho mỗi người chơi (phút)")


class MatchJoinRequest(BaseModel):
    join_code: str

class JoinByCodeRequest(BaseModel):
    room_code: str = Field(..., min_length=6, max_length=6, description="Mã bàn 6 ký tự")


class MoveRequest(BaseModel):
    x: conint(ge=0)
    y: conint(ge=0)
    move_number: int
    color: Literal["B", "W"]


class PassRequest(BaseModel):
    move_number: int
    color: Literal["B", "W"]


class SGFImportRequest(BaseModel):
    sgf_content: str = Field(..., description="SGF format string")


class MatchState(BaseModel):
    size: int
    to_move: Literal["B", "W"]
    moves: Sequence[tuple[int, int]]
    board_position: dict[str, Literal["B", "W"]] | None = None  # Current board state (key: "x,y", value: "B" or "W")
    prisoners_black: int = 0  # Số quân đen bị bắt
    prisoners_white: int = 0  # Số quân trắng bị bắt
    black_time_remaining_seconds: int | None = None  # Thời gian còn lại của Black (giây)
    white_time_remaining_seconds: int | None = None  # Thời gian còn lại của White (giây)


class MatchReadyRequest(BaseModel):
    """Request để set ready status cho match."""
    ready: bool = Field(..., description="True nếu sẵn sàng, False nếu chưa sẵn sàng")


class MatchResponse(MatchBase):
    black_player_id: UUID | None = None
    white_player_id: UUID | None = None
    black_player_username: str | None = None  # Tên người chơi Black
    white_player_username: str | None = None  # Tên người chơi White
    premium_analysis_id: str | None = None
    room_code: str | None = None  # Mã bàn cho PvP matches
    black_elo_change: int | None = None  # ELO change của Black player (chỉ cho PvP)
    white_elo_change: int | None = None  # ELO change của White player (chỉ cho PvP)
    user_elo_change: int | None = None  # ELO change của user hiện tại (tính toán dựa trên user là Black hay White)
    user_color: Literal["B", "W"] | None = None  # Màu quân của user hiện tại trong match này
    black_ready: bool = False  # Black player đã sẵn sàng
    white_ready: bool = False  # White player đã sẵn sàng
    state: MatchState | None = None

