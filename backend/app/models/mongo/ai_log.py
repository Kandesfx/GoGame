"""Model cho collection ai_logs."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class AILogDocument(BaseModel):
    id: str = Field(alias="_id")
    match_id: str
    move_number: int
    engine: str
    config: Dict[str, Any] | None = None
    stats: Dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True

