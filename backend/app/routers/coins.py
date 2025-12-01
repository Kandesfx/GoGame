"""Router hệ thống coins."""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user, get_coin_service, get_payment_service
from ..models.sql import user as user_models
from ..schemas import coins as coin_schema
from ..services.coin_service import CoinService
from ..services.payment_service import PaymentService, COIN_PACKAGES

router = APIRouter()


@router.get("/balance", response_model=coin_schema.CoinBalance)
def get_balance(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[CoinService, Depends(get_coin_service)],
):
    return coin_schema.CoinBalance(**service.get_balance(current_user))


@router.get("/history", response_model=List[coin_schema.CoinTransaction])
def get_history(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[CoinService, Depends(get_coin_service)],
):
    transactions = service.list_transactions(current_user)
    return [
        coin_schema.CoinTransaction(
            id=transaction.id,
            user_id=transaction.user_id,
            amount=transaction.amount,
            type=transaction.type,
            source=transaction.source,
            created_at=transaction.created_at,
        )
        for transaction in transactions
    ]


@router.get("/packages", response_model=coin_schema.CoinPackagesResponse)
def get_packages():
    """Lấy danh sách các gói coin có sẵn."""
    packages = [
        coin_schema.CoinPackage(
            id=package_id,
            name=package_id.capitalize(),
            coins=package["coins"],
            bonus_coins=package.get("bonus", 0),
            price_usd=package["price_usd"],
            description=f"{package['coins']} coins" + (f" + {package.get('bonus', 0)} bonus" if package.get("bonus", 0) > 0 else ""),
        )
        for package_id, package in COIN_PACKAGES.items()
    ]
    return coin_schema.CoinPackagesResponse(packages=packages)


@router.post("/purchase", response_model=dict)
def purchase_coins(
    payload: coin_schema.CoinPurchaseRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
):
    """Mua coins từ package."""
    try:
        result = payment_service.purchase_coins(current_user, payload.package_id, payload.payment_token)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/daily-bonus", response_model=coin_schema.DailyBonusResponse)
def claim_daily_bonus(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[CoinService, Depends(get_coin_service)],
):
    """Claim daily login bonus."""
    result = service.claim_daily_bonus(current_user)
    return coin_schema.DailyBonusResponse(**result)


@router.post("/earn", response_model=dict)
def earn_coins(
    payload: coin_schema.EarnCoinsRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[CoinService, Depends(get_coin_service)],
):
    """Earn coins từ các hành động (complete game, win game, etc.)."""
    try:
        result = service.earn_coins(current_user, payload.action, payload.amount)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

