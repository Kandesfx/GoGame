"""Service cho statistics và Elo rating."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from ..models.sql import match as match_model
from ..models.sql import user as user_model

logger = logging.getLogger(__name__)

# Elo rating constants
K_FACTOR = 32  # Standard K-factor for Elo
INITIAL_RATING = 1500


def calculate_expected_score(rating_a: int, rating_b: int) -> float:
    """Tính expected score cho player A.
    
    Args:
        rating_a: Elo rating của player A
        rating_b: Elo rating của player B
    
    Returns:
        Expected score (0.0 - 1.0)
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def calculate_elo_change(rating: int, opponent_rating: int, actual_score: float) -> int:
    """Tính Elo rating change.
    
    Args:
        rating: Current rating
        opponent_rating: Opponent's rating
        actual_score: 1.0 for win, 0.5 for draw, 0.0 for loss
    
    Returns:
        Rating change (có thể âm)
    """
    expected_score = calculate_expected_score(rating, opponent_rating)
    change = int(K_FACTOR * (actual_score - expected_score))
    return change


class StatisticsService:
    """Service cho user statistics và Elo rating."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_user_statistics(self, user_id: UUID) -> Dict[str, any]:
        """Lấy statistics của user.
        
        Args:
            user_id: User ID
        
        Returns:
            Dict với statistics
        """
        user = self.db.get(user_model.User, str(user_id))
        if not user:
            raise ValueError("User not found")

        # Get all completed matches
        matches = (
            self.db.query(match_model.Match)
            .filter(
                and_(
                    or_(
                        match_model.Match.black_player_id == str(user_id),
                        match_model.Match.white_player_id == str(user_id),
                    ),
                    match_model.Match.result.isnot(None),
                )
            )
            .all()
        )

        # Calculate statistics
        total_matches = len(matches)
        wins = 0
        losses = 0
        draws = 0
        
        # Statistics by AI level (1=dễ, 2=trung bình, 3=khó, 4=siêu khó)
        ai_stats = {
            1: {"wins": 0, "losses": 0, "draws": 0, "matches": 0},  # Dễ
            2: {"wins": 0, "losses": 0, "draws": 0, "matches": 0},  # Trung bình
            3: {"wins": 0, "losses": 0, "draws": 0, "matches": 0},  # Khó
            4: {"wins": 0, "losses": 0, "draws": 0, "matches": 0},  # Siêu khó
        }
        
        # Statistics for PvP (online matches, no AI)
        wins_vs_player = 0
        losses_vs_player = 0
        draws_vs_player = 0
        matches_vs_player = 0

        for match in matches:
            if not match.result:
                continue

            # Determine if user won and match type
            is_black = match.black_player_id == str(user_id)
            is_ai_match = match.ai_level is not None
            ai_level = match.ai_level
            result = match.result.upper()

            if "B+" in result:
                if is_black:
                    wins += 1
                    if is_ai_match and ai_level in ai_stats:
                        ai_stats[ai_level]["wins"] += 1
                        ai_stats[ai_level]["matches"] += 1
                    elif not is_ai_match:
                        wins_vs_player += 1
                        matches_vs_player += 1
                else:
                    losses += 1
                    if is_ai_match and ai_level in ai_stats:
                        ai_stats[ai_level]["losses"] += 1
                        ai_stats[ai_level]["matches"] += 1
                    elif not is_ai_match:
                        losses_vs_player += 1
                        matches_vs_player += 1
            elif "W+" in result:
                if is_black:
                    losses += 1
                    if is_ai_match and ai_level in ai_stats:
                        ai_stats[ai_level]["losses"] += 1
                        ai_stats[ai_level]["matches"] += 1
                    elif not is_ai_match:
                        losses_vs_player += 1
                        matches_vs_player += 1
                else:
                    wins += 1
                    if is_ai_match and ai_level in ai_stats:
                        ai_stats[ai_level]["wins"] += 1
                        ai_stats[ai_level]["matches"] += 1
                    elif not is_ai_match:
                        wins_vs_player += 1
                        matches_vs_player += 1
            elif result in ("DRAW", "JIGO"):
                draws += 1
                if is_ai_match and ai_level in ai_stats:
                    ai_stats[ai_level]["draws"] += 1
                    ai_stats[ai_level]["matches"] += 1
                elif not is_ai_match:
                    draws_vs_player += 1
                    matches_vs_player += 1

        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0.0
        
        # Calculate win rates by AI level
        win_rate_ai_easy = (ai_stats[1]["wins"] / ai_stats[1]["matches"] * 100) if ai_stats[1]["matches"] > 0 else 0.0
        win_rate_ai_medium = (ai_stats[2]["wins"] / ai_stats[2]["matches"] * 100) if ai_stats[2]["matches"] > 0 else 0.0
        win_rate_ai_hard = (ai_stats[3]["wins"] / ai_stats[3]["matches"] * 100) if ai_stats[3]["matches"] > 0 else 0.0
        win_rate_ai_super_hard = (ai_stats[4]["wins"] / ai_stats[4]["matches"] * 100) if ai_stats[4]["matches"] > 0 else 0.0
        
        # Calculate win rate for PvP
        win_rate_vs_player = (wins_vs_player / matches_vs_player * 100) if matches_vs_player > 0 else 0.0

        # Get recent matches
        recent_matches = (
            self.db.query(match_model.Match)
            .filter(
                or_(
                    match_model.Match.black_player_id == str(user_id),
                    match_model.Match.white_player_id == str(user_id),
                )
            )
            .order_by(match_model.Match.started_at.desc())
            .limit(10)
            .all()
        )

        return {
            "user_id": str(user_id),
            "username": user.username,
            "elo_rating": user.elo_rating,
            "total_matches": total_matches,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(win_rate, 2),
            # Statistics by AI level
            "wins_ai_easy": ai_stats[1]["wins"],
            "losses_ai_easy": ai_stats[1]["losses"],
            "draws_ai_easy": ai_stats[1]["draws"],
            "matches_ai_easy": ai_stats[1]["matches"],
            "win_rate_ai_easy": round(win_rate_ai_easy, 2),
            "wins_ai_medium": ai_stats[2]["wins"],
            "losses_ai_medium": ai_stats[2]["losses"],
            "draws_ai_medium": ai_stats[2]["draws"],
            "matches_ai_medium": ai_stats[2]["matches"],
            "win_rate_ai_medium": round(win_rate_ai_medium, 2),
            "wins_ai_hard": ai_stats[3]["wins"],
            "losses_ai_hard": ai_stats[3]["losses"],
            "draws_ai_hard": ai_stats[3]["draws"],
            "matches_ai_hard": ai_stats[3]["matches"],
            "win_rate_ai_hard": round(win_rate_ai_hard, 2),
            "wins_ai_super_hard": ai_stats[4]["wins"],
            "losses_ai_super_hard": ai_stats[4]["losses"],
            "draws_ai_super_hard": ai_stats[4]["draws"],
            "matches_ai_super_hard": ai_stats[4]["matches"],
            "win_rate_ai_super_hard": round(win_rate_ai_super_hard, 2),
            # Statistics for PvP (online matches)
            "wins_vs_player": wins_vs_player,
            "losses_vs_player": losses_vs_player,
            "draws_vs_player": draws_vs_player,
            "matches_vs_player": matches_vs_player,
            "win_rate_vs_player": round(win_rate_vs_player, 2),
            "recent_matches": [
                {
                    "match_id": str(match.id),
                    "result": match.result,
                    "started_at": match.started_at.isoformat() if match.started_at else None,
                }
                for match in recent_matches
            ],
        }

    def update_elo_ratings(self, match: match_model.Match) -> None:
        """Update Elo ratings sau khi match kết thúc.
        
        Args:
            match: Match object với result đã set
        """
        if not match.result:
            logger.warning(f"Match {match.id} has no result, skipping Elo update")
            return

        # Chỉ update cho PvP matches (không có AI)
        if match.ai_level:
            logger.debug(f"Match {match.id} is AI match, skipping Elo update")
            return

        if not match.black_player_id or not match.white_player_id:
            # Chỉ log warning nếu không phải AI match (AI match có white_player_id=None là bình thường)
            if not match.ai_level:
                logger.warning(f"Match {match.id} missing players, skipping Elo update")
            return

        black_user = self.db.get(user_model.User, match.black_player_id)
        white_user = self.db.get(user_model.User, match.white_player_id)

        if not black_user or not white_user:
            logger.warning(f"Users not found for match {match.id}")
            return

        # Determine actual scores
        result = match.result.upper()
        if "B+" in result:
            black_score = 1.0
            white_score = 0.0
        elif "W+" in result:
            black_score = 0.0
            white_score = 1.0
        elif result in ("DRAW", "JIGO"):
            black_score = 0.5
            white_score = 0.5
        else:
            logger.warning(f"Unknown result format: {match.result}")
            return

        # Calculate rating changes
        black_change = calculate_elo_change(black_user.elo_rating, white_user.elo_rating, black_score)
        white_change = calculate_elo_change(white_user.elo_rating, black_user.elo_rating, white_score)

        # Lưu ELO change vào match
        match.black_elo_change = black_change
        match.white_elo_change = white_change

        # Update ratings
        black_user.elo_rating += black_change
        white_user.elo_rating += white_change

        # Ensure ratings don't go below 0
        black_user.elo_rating = max(0, black_user.elo_rating)
        white_user.elo_rating = max(0, white_user.elo_rating)

        self.db.commit()

        logger.info(
            f"Elo updated for match {match.id}: "
            f"Black {black_user.username} {black_user.elo_rating - black_change} -> {black_user.elo_rating} "
            f"({black_change:+d}), "
            f"White {white_user.username} {white_user.elo_rating - white_change} -> {white_user.elo_rating} "
            f"({white_change:+d})"
        )

    def get_leaderboard(self, limit: int = 100) -> List[Dict[str, any]]:
        """Lấy leaderboard (top players by Elo).
        
        Args:
            limit: Số lượng players trả về
        
        Returns:
            List of user statistics
        """
        users = (
            self.db.query(user_model.User)
            .order_by(user_model.User.elo_rating.desc())
            .limit(limit)
            .all()
        )

        leaderboard = []
        for rank, user in enumerate(users, 1):
            stats = self.get_user_statistics(UUID(user.id))
            leaderboard.append(
                {
                    "rank": rank,
                    "user_id": str(user.id),
                    "username": user.username,
                    "display_name": user.display_name,
                    "elo_rating": user.elo_rating,
                    "total_matches": stats["total_matches"],
                    "win_rate": stats["win_rate"],
                }
            )

        return leaderboard

