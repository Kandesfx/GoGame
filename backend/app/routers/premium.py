"""Router tính năng premium."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user, get_match_service, get_payment_service, get_premium_service
from ..models.sql import user as user_models
from ..schemas import premium as premium_schema
from ..services.match_service import MatchService
from ..services.payment_service import PaymentService, PREMIUM_PLANS
from ..services.premium_service import PremiumService

router = APIRouter()


@router.post("/hint", response_model=dict)
async def request_hint(
    payload: premium_schema.PremiumHintRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.get_match(payload.match_id)
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    return await premium_service.create_hint(current_user, match, payload, match_service)


@router.post("/analysis", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def request_analysis(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.get_match(match_id)
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    return await premium_service.create_analysis(current_user, match, match_service)


@router.post("/review", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def request_review(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.get_match(match_id)
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    return await premium_service.create_review(current_user, match, match_service)


@router.get("/requests/{request_id}", response_model=dict)
async def get_premium_request(
    request_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)],
):
    report = await premium_service.get_request(request_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    return report


@router.get("/cache/stats", response_model=dict)
async def get_cache_stats(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)],
):
    """Lấy thống kê evaluation cache (dùng cho monitoring/debugging)."""
    # TODO: Có thể thêm permission check (chỉ admin)
    stats = premium_service.eval_cache.get_stats()
    return stats


@router.get("/subscription/status", response_model=premium_schema.PremiumSubscriptionStatus | None)
def get_subscription_status(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
):
    """Lấy trạng thái premium subscription của user."""
    status_info = payment_service.get_subscription_status(current_user)
    if not status_info:
        return None
    return premium_schema.PremiumSubscriptionStatus(**status_info)


@router.get("/subscription/plans", response_model=premium_schema.PremiumPlansResponse)
def get_subscription_plans():
    """Lấy danh sách các gói premium subscription có sẵn."""
    plans = [
        premium_schema.PremiumPlan(
            id=plan_id,
            name=plan_id.capitalize(),
            price_usd=plan_config["price_usd"],
            duration_days=plan_config["duration_days"],
            bonus_coins=plan_config.get("bonus_coins", 0),
            description=f"{plan_config['duration_days']} days" + (f" + {plan_config.get('bonus_coins', 0)} bonus coins" if plan_config.get("bonus_coins", 0) > 0 else ""),
        )
        for plan_id, plan_config in PREMIUM_PLANS.items()
    ]
    return premium_schema.PremiumPlansResponse(plans=plans)


@router.post("/subscription/subscribe", response_model=premium_schema.PremiumSubscriptionResponse)
def subscribe_premium(
    payload: premium_schema.PremiumSubscriptionRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
):
    """Đăng ký premium subscription."""
    try:
        result = payment_service.subscribe_premium(current_user, payload.plan, payload.payment_token)
        return premium_schema.PremiumSubscriptionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/subscription/cancel", response_model=dict)
def cancel_subscription(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    payment_service: Annotated[PaymentService, Depends(get_payment_service)],
):
    """Hủy premium subscription (vẫn dùng đến hết hạn)."""
    try:
        result = payment_service.cancel_subscription(current_user)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

