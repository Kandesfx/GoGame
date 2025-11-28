"""Schemas cho statistics."""

from typing import List
from uuid import UUID

from pydantic import BaseModel


class MatchSummary(BaseModel):
    match_id: UUID
    result: str | None
    started_at: str | None


class UserStatistics(BaseModel):
    user_id: UUID
    username: str
    elo_rating: int
    total_matches: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    # Statistics by AI level
    wins_ai_easy: int = 0
    losses_ai_easy: int = 0
    draws_ai_easy: int = 0
    matches_ai_easy: int = 0
    win_rate_ai_easy: float = 0.0
    wins_ai_medium: int = 0
    losses_ai_medium: int = 0
    draws_ai_medium: int = 0
    matches_ai_medium: int = 0
    win_rate_ai_medium: float = 0.0
    wins_ai_hard: int = 0
    losses_ai_hard: int = 0
    draws_ai_hard: int = 0
    matches_ai_hard: int = 0
    win_rate_ai_hard: float = 0.0
    wins_ai_super_hard: int = 0
    losses_ai_super_hard: int = 0
    draws_ai_super_hard: int = 0
    matches_ai_super_hard: int = 0
    win_rate_ai_super_hard: float = 0.0
    # Statistics for PvP (online matches)
    wins_vs_player: int = 0
    losses_vs_player: int = 0
    draws_vs_player: int = 0
    matches_vs_player: int = 0
    win_rate_vs_player: float = 0.0
    recent_matches: List[MatchSummary]


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: UUID
    username: str
    display_name: str | None
    elo_rating: int
    total_matches: int
    win_rate: float

