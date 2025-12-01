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
    payment_token: str | None = None  # Payment token từ payment gateway


class CoinPackage(BaseModel):
    id: str
    name: str
    coins: int
    bonus_coins: int
    price_usd: float
    description: str | None = None


class CoinPackagesResponse(BaseModel):
    packages: list[CoinPackage]


class DailyBonusResponse(BaseModel):
    success: bool
    coins_earned: int | None = None
    new_balance: int | None = None
    message: str | None = None
    transaction_id: UUID | None = None


class EarnCoinsRequest(BaseModel):
    action: Literal["complete_game", "win_game", "rank_up", "achievement", "watch_ad"]
    amount: int | None = None  # Optional custom amount

