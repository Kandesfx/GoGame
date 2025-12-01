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


class PremiumSubscriptionStatus(BaseModel):
    id: UUID
    plan: Literal["monthly", "yearly"]
    status: Literal["active", "expired", "cancelled"]
    is_active: bool
    started_at: str
    expires_at: str
    cancelled_at: str | None = None


class PremiumSubscriptionRequest(BaseModel):
    plan: Literal["monthly", "yearly"]
    payment_token: str | None = None  # Payment token từ payment gateway


class PremiumSubscriptionResponse(BaseModel):
    success: bool
    subscription_id: UUID
    plan: str
    expires_at: str
    bonus_coins: int
    new_balance: int


class PremiumPlan(BaseModel):
    id: str
    name: str
    price_usd: float
    duration_days: int
    bonus_coins: int
    description: str | None = None


class PremiumPlansResponse(BaseModel):
    plans: list[PremiumPlan]
