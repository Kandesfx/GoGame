"""Service quản lý coin."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from ..models.sql import coin_transaction as coin_model
from ..models.sql import user as user_model


# Coin earning configuration
COIN_EARN_RULES = {
    "daily_login": 10,
    "complete_game": 5,
    "win_game": 10,
    "rank_up": 50,
    "achievement": 20,
    "watch_ad": 5,  # Optional
}


class CoinService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_balance(self, user: user_model.User) -> dict:
        """Lấy số dư coins và trạng thái daily bonus."""
        # Check if user has claimed daily bonus today
        today = datetime.now(tz=timezone.utc).date()
        last_daily = (
            self.db.query(coin_model.CoinTransaction)
            .filter(coin_model.CoinTransaction.user_id == user.id)
            .filter(coin_model.CoinTransaction.type == "earn")
            .filter(coin_model.CoinTransaction.source == "daily_login")
            .order_by(coin_model.CoinTransaction.created_at.desc())
            .first()
        )

        has_daily_bonus = True
        if last_daily:
            last_daily_date = last_daily.created_at.date()
            has_daily_bonus = last_daily_date < today

        return {"coins": user.coins, "has_daily_bonus": has_daily_bonus}

    def list_transactions(self, user: user_model.User, limit: int = 50) -> List[coin_model.CoinTransaction]:
        return (
            self.db.query(coin_model.CoinTransaction)
            .filter(coin_model.CoinTransaction.user_id == user.id)
            .order_by(coin_model.CoinTransaction.created_at.desc())
            .limit(limit)
            .all()
        )

    def add_transaction(self, user: user_model.User, amount: int, txn_type: str, source: str | None = None):
        """Thêm transaction và cập nhật balance."""
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

    def claim_daily_bonus(self, user: user_model.User) -> dict:
        """
        Claim daily login bonus.
        
        Returns:
            Dict với thông tin transaction hoặc None nếu đã claim hôm nay
        """
        balance_info = self.get_balance(user)
        if not balance_info["has_daily_bonus"]:
            return {"success": False, "message": "Đã nhận daily bonus hôm nay rồi"}

        coins = COIN_EARN_RULES["daily_login"]
        transaction = self.add_transaction(user, coins, "earn", source="daily_login")

        return {
            "success": True,
            "coins_earned": coins,
            "new_balance": user.coins,
            "transaction_id": transaction.id,
        }

    def earn_coins(self, user: user_model.User, action: str, amount: int | None = None) -> dict:
        """
        Earn coins từ các hành động (complete game, win game, etc.).
        
        Args:
            user: User object
            action: Action type (complete_game, win_game, rank_up, achievement, watch_ad)
            amount: Custom amount (optional, nếu không dùng sẽ dùng từ COIN_EARN_RULES)
        
        Returns:
            Dict với thông tin transaction
        """
        if action not in COIN_EARN_RULES and amount is None:
            raise ValueError(f"Invalid action: {action}")

        coins = amount if amount is not None else COIN_EARN_RULES[action]
        transaction = self.add_transaction(user, coins, "earn", source=action)

        return {
            "success": True,
            "coins_earned": coins,
            "new_balance": user.coins,
            "transaction_id": transaction.id,
        }

