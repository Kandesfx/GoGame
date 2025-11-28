"""Model cho collection games."""

from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


class MoveRecord(BaseModel):
    number: int
    color: str
    position: Optional[Tuple[int, int]] = None
    policy: Optional[Sequence[float]] = None
    value: Optional[float] = None


class AnalysisRecord(BaseModel):
    winrate_curve: Optional[List[float]] = None
    key_mistakes: Optional[List[dict]] = None
    comments: Optional[List[dict]] = None


class GameDocument(BaseModel):
    id: str = Field(alias="_id")
    match_id: str
    board_size: int = 9
    sgf: Optional[str] = None
    moves: List[MoveRecord] = Field(default_factory=list)
    analysis: Optional[AnalysisRecord] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True

