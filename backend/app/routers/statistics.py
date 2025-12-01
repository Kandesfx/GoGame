"""API endpoints cho statistics và leaderboard."""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from uuid import UUID

from ..dependencies import get_current_user, get_statistics_service
from ..models.sql import user as user_models
from ..schemas import statistics as statistics_schema
from ..services.statistics_service import StatisticsService

router = APIRouter()


@router.get("/me", response_model=statistics_schema.UserStatistics)
def get_my_statistics(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    statistics_service: Annotated[StatisticsService, Depends(get_statistics_service)],
):
    """Lấy statistics của current user."""
    try:
        return statistics_service.get_user_statistics(UUID(current_user.id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/user/{user_id}", response_model=statistics_schema.UserStatistics)
def get_user_statistics(
    user_id: UUID,
    statistics_service: Annotated[StatisticsService, Depends(get_statistics_service)],
):
    """Lấy statistics của một user (public)."""
    try:
        return statistics_service.get_user_statistics(user_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/leaderboard", response_model=List[statistics_schema.LeaderboardEntry])
def get_leaderboard(
    statistics_service: Annotated[StatisticsService, Depends(get_statistics_service)],
    limit: Annotated[int, Query(ge=1, le=1000, description="Number of entries")] = 100,
):
    """Lấy leaderboard (top players by Elo)."""
    return statistics_service.get_leaderboard(limit=limit)

