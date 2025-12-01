"""Service quản lý coin."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from ..models.sql import coin_transaction as coin_model
from ..models.sql import user as user_model


class CoinService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_balance(self, user: user_model.User) -> dict:
        return {"coins": user.coins, "has_daily_bonus": True}

    def list_transactions(self, user: user_model.User, limit: int = 50) -> List[coin_model.CoinTransaction]:
        return (
            self.db.query(coin_model.CoinTransaction)
            .filter(coin_model.CoinTransaction.user_id == user.id)
            .order_by(coin_model.CoinTransaction.created_at.desc())
            .limit(limit)
            .all()
        )

    def add_transaction(self, user: user_model.User, amount: int, txn_type: str, source: str | None = None):
        user.coins += amount
        transaction = coin_model.CoinTransaction(
            id=str(uuid4()),
            user_id=user.id,
            amount=amount,
            type=txn_type,
            source=source,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(user)
        self.db.refresh(transaction)
        return transaction

