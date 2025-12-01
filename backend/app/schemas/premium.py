"""Schema cho tính năng premium."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, conint


class PremiumFeature(str):
    HINT = "hint"
    ANALYSIS = "analysis"
    REVIEW = "review"


class PremiumRequestBase(BaseModel):
    id: UUID
    match_id: UUID
    user_id: UUID
    feature: str
    cost: int
    status: Literal["pending", "completed", "failed"]
    created_at: datetime
    completed_at: datetime | None = None


class PremiumHintRequest(BaseModel):
    match_id: UUID
    top_k: conint(ge=1, le=5) = 3


class PremiumAsyncResponse(BaseModel):
    request_id: UUID
    status: Literal["pending", "completed"]

