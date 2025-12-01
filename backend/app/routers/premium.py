"""Router tính năng premium."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user, get_match_service, get_premium_service
from ..models.sql import user as user_models
from ..schemas import premium as premium_schema
from ..services.match_service import MatchService
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

