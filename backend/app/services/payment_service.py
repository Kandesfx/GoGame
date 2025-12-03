"""Service xử lý thanh toán (mock implementation)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from ..models.sql import premium_subscription as premium_sub_model
from ..models.sql import user as user_model
from .coin_service import CoinService

logger = logging.getLogger(__name__)


# Coin packages configuration
COIN_PACKAGES: Dict[str, Dict[str, any]] = {
    "starter": {"coins": 100, "price_usd": 0.99, "bonus": 0},
    "basic": {"coins": 500, "price_usd": 3.99, "bonus": 0},
    "standard": {"coins": 1200, "price_usd": 8.99, "bonus": 200},  # 20% bonus
    "premium": {"coins": 3000, "price_usd": 19.99, "bonus": 1500},  # 50% bonus
    "ultimate": {"coins": 10000, "price_usd": 49.99, "bonus": 10000},  # 100% bonus
}

# Premium subscription plans
PREMIUM_PLANS: Dict[str, Dict[str, any]] = {
    "monthly": {"price_usd": 4.99, "duration_days": 30, "bonus_coins": 500},
    "yearly": {"price_usd": 49.99, "duration_days": 365, "bonus_coins": 6000},  # ~17% discount
}


class PaymentService:
    """Service xử lý thanh toán (mock - chưa tích hợp payment gateway thật)."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.coin_service = CoinService(db)

    def purchase_coins(self, user: user_model.User, package_id: str, payment_token: Optional[str] = None) -> Dict:
        """
        Mua coins từ package.
        
        Args:
            user: User object
            package_id: ID của package (starter, basic, standard, premium, ultimate)
            payment_token: Payment token từ payment gateway (mock - không dùng)
        
        Returns:
            Dict với thông tin transaction
        """
        if package_id not in COIN_PACKAGES:
            raise ValueError(f"Invalid package_id: {package_id}")

        package = COIN_PACKAGES[package_id]
        coins = package["coins"] + package.get("bonus", 0)

        # Mock payment processing (trong thực tế sẽ gọi payment gateway)
        logger.info(f"Processing payment for user {user.id}: package={package_id}, coins={coins}")

        # Trong production, cần verify payment_token với payment gateway
        # if not self._verify_payment(payment_token, package["price_usd"]):
        #     raise ValueError("Payment verification failed")

        # Add coins to user
        transaction = self.coin_service.add_transaction(
            user, coins, "purchase", source=f"package_{package_id}"
        )

        return {
            "success": True,
            "transaction_id": transaction.id,
            "coins_added": coins,
            "base_coins": package["coins"],
            "bonus_coins": package.get("bonus", 0),
            "new_balance": user.coins,
        }

    def subscribe_premium(self, user: user_model.User, plan: str, payment_token: Optional[str] = None) -> Dict:
        """
        Đăng ký premium subscription.
        
        Args:
            user: User object
            plan: "monthly" hoặc "yearly"
            payment_token: Payment token từ payment gateway (mock - không dùng)
        
        Returns:
            Dict với thông tin subscription
        """
        if plan not in PREMIUM_PLANS:
            raise ValueError(f"Invalid plan: {plan}. Must be 'monthly' or 'yearly'")

        plan_config = PREMIUM_PLANS[plan]

        # Mock payment processing
        logger.info(f"Processing premium subscription for user {user.id}: plan={plan}")

        # Trong production, cần verify payment_token với payment gateway
        # if not self._verify_payment(payment_token, plan_config["price_usd"]):
        #     raise ValueError("Payment verification failed")

        # Check if user already has ANY subscription (active, cancelled, or expired)
        # QUAN TRỌNG: Check tất cả subscriptions, không chỉ active, để tránh duplicate key
        existing_sub = (
            self.db.query(premium_sub_model.PremiumSubscription)
            .filter(premium_sub_model.PremiumSubscription.user_id == user.id)
            .first()
        )

        now = datetime.now(tz=timezone.utc)
        expires_at = now + timedelta(days=plan_config["duration_days"])

        if existing_sub:
            # Extend existing subscription
            existing_sub.plan = plan
            existing_sub.expires_at = expires_at
            existing_sub.status = "active"
            existing_sub.updated_at = now
            if existing_sub.cancelled_at:
                existing_sub.cancelled_at = None
            self.db.commit()
            self.db.refresh(existing_sub)
            subscription = existing_sub
        else:
            # Create new subscription
            subscription = premium_sub_model.PremiumSubscription(
                id=str(uuid4()),
                user_id=user.id,
                plan=plan,
                status="active",
                started_at=now,
                expires_at=expires_at,
                created_at=now,
                updated_at=now,
            )
            self.db.add(subscription)
            self.db.commit()
            self.db.refresh(subscription)

        # Add bonus coins
        if plan_config.get("bonus_coins", 0) > 0:
            self.coin_service.add_transaction(
                user, plan_config["bonus_coins"], "earn", source=f"premium_subscription_{plan}"
            )

        return {
            "success": True,
            "subscription_id": subscription.id,
            "plan": plan,
            "expires_at": subscription.expires_at.isoformat(),
            "bonus_coins": plan_config.get("bonus_coins", 0),
            "new_balance": user.coins,
        }

    def cancel_subscription(self, user: user_model.User) -> Dict:
        """
        Hủy premium subscription (vẫn dùng đến hết hạn).
        
        Args:
            user: User object
        
        Returns:
            Dict với thông tin cancellation
        """
        subscription = (
            self.db.query(premium_sub_model.PremiumSubscription)
            .filter(premium_sub_model.PremiumSubscription.user_id == user.id)
            .filter(premium_sub_model.PremiumSubscription.status == "active")
            .first()
        )

        if not subscription:
            raise ValueError("No active subscription found")

        subscription.status = "cancelled"
        subscription.cancelled_at = datetime.now(tz=timezone.utc)
        subscription.updated_at = datetime.now(tz=timezone.utc)
        self.db.commit()
        self.db.refresh(subscription)

        return {
            "success": True,
            "subscription_id": subscription.id,
            "expires_at": subscription.expires_at.isoformat(),
            "message": "Subscription cancelled. You can still use premium features until expiration.",
        }

    def get_subscription_status(self, user: user_model.User) -> Optional[Dict]:
        """
        Lấy trạng thái premium subscription của user.
        
        Args:
            user: User object
        
        Returns:
            Dict với thông tin subscription hoặc None nếu không có
        """
        subscription = (
            self.db.query(premium_sub_model.PremiumSubscription)
            .filter(premium_sub_model.PremiumSubscription.user_id == user.id)
            .order_by(premium_sub_model.PremiumSubscription.created_at.desc())
            .first()
        )

        if not subscription:
            return None

        now = datetime.now(tz=timezone.utc)
        is_active = subscription.status == "active" and subscription.expires_at > now

        return {
            "id": subscription.id,
            "plan": subscription.plan,
            "status": subscription.status,
            "is_active": is_active,
            "started_at": subscription.started_at.isoformat(),
            "expires_at": subscription.expires_at.isoformat(),
            "cancelled_at": subscription.cancelled_at.isoformat() if subscription.cancelled_at else None,
        }

    def _verify_payment(self, payment_token: str, amount: float) -> bool:
        """
        Verify payment với payment gateway (mock implementation).
        
        Trong production, cần tích hợp với payment gateway thật như:
        - Stripe
        - PayPal
        - VNPay (cho Việt Nam)
        - etc.
        """
        # Mock: luôn return True
        # Trong production:
        # - Gọi API của payment gateway để verify
        # - Check amount, currency, etc.
        return True

