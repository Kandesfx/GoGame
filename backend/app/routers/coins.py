"""Router hệ thống coins."""

from typing import Annotated, List

from fastapi import APIRouter, Depends

from ..dependencies import get_current_user, get_coin_service
from ..models.sql import user as user_models
from ..schemas import coins as coin_schema
from ..services.coin_service import CoinService

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


@router.post("/purchase", response_model=coin_schema.CoinTransaction)
def purchase(
    payload: coin_schema.CoinPurchaseRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[CoinService, Depends(get_coin_service)],
):
    transaction = service.add_transaction(current_user, payload.amount, "purchase", source=payload.package_id)
    return coin_schema.CoinTransaction(
        id=transaction.id,
        user_id=transaction.user_id,
        amount=transaction.amount,
        type=transaction.type,
        source=transaction.source,
        created_at=transaction.created_at,
    )

