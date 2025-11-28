"""Schema cho hệ thống coins."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, conint


class CoinBalance(BaseModel):
    coins: int
    has_daily_bonus: bool = True


class CoinTransaction(BaseModel):
    id: UUID
    user_id: UUID
    amount: int
    type: Literal["purchase", "earn", "spend"]
    source: str | None = None
    created_at: datetime


class CoinPurchaseRequest(BaseModel):
    package_id: str
    amount: conint(gt=0)

