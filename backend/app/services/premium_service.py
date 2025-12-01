"""Service cho tính năng premium."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import gogame_py as go
except ImportError:
    go = None  # type: ignore
    logging.warning("gogame_py module not found. Premium features will use fallback.")

from uuid import UUID, uuid4

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from ..config import Settings
from ..models.sql import match as match_model
from ..models.sql import premium_request as premium_model
from ..models.sql import user as user_model
from ..schemas import premium as premium_schema
from ..utils.evaluation_cache import get_evaluation_cache
from .coin_service import CoinService
from .match_service import MatchService

logger = logging.getLogger(__name__)


class PremiumService:
    HINT_COST = 10
    ANALYSIS_COST = 20
    REVIEW_COST = 30

    def __init__(self, db: Session, mongo_db: AsyncIOMotorDatabase, settings: Settings) -> None:
        self.db = db
        self.mongo_db = mongo_db
        self.settings = settings
        self.coin_service = CoinService(db)
        self.ai_player = go.AIPlayer() if go else None
        # Evaluation cache để tối ưu performance
        self.eval_cache = get_evaluation_cache(
            max_size=settings.eval_cache_max_size, ttl_seconds=settings.eval_cache_ttl_seconds
        )

    def _ensure_balance(self, user: user_model.User, cost: int):
        if user.coins < cost:
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Không đủ coins")

    def _create_request(
        self,
        user: user_model.User,
        match: match_model.Match,
        feature: str,
        cost: int,
        status_value: str = "pending",
    ) -> premium_model.PremiumRequest:
        request = premium_model.PremiumRequest(
            id=str(uuid4()),
            user_id=user.id,
            match_id=match.id,
            feature=feature,
            cost=cost,
            status=status_value,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.db.add(request)
        self.db.commit()
        self.db.refresh(request)
        return request

    async def create_hint(
        self,
        user: user_model.User,
        match: match_model.Match,
        payload: premium_schema.PremiumHintRequest,
        match_service: MatchService,
    ) -> Dict[str, Any]:
        """Tạo gợi ý nước đi sử dụng MCTS hoặc Policy Network."""
        self._ensure_balance(user, self.HINT_COST)
        self.coin_service.add_transaction(user, -self.HINT_COST, "spend", source="premium_hint")

        request = self._create_request(user, match, feature="hint", cost=self.HINT_COST, status_value="completed")

        hints: List[Dict[str, Any]] = []

        if go and self.ai_player:
            try:
                # Load board state
                board = await match_service._get_or_create_board(match)

                # Sử dụng MCTS để tìm top moves
                loop = asyncio.get_event_loop()

                # Tạo MCTS engine với nhiều playouts để có nhiều options
                mcts_config = go.MCTSConfig(
                    num_playouts=2000,
                    time_limit_seconds=5.0,
                    use_heuristics=True,
                )
                mcts_engine = go.MCTSEngine(mcts_config)

                # Chạy MCTS search để lấy top moves
                search_result = await loop.run_in_executor(
                    None,
                    lambda: mcts_engine.search(board, board.current_player()),
                )

                # Lấy top moves từ search result
                if hasattr(search_result, "top_moves") and search_result.top_moves:
                    for move_stats in search_result.top_moves[: payload.top_k]:
                        move = move_stats.move
                        hints.append(
                            {
                                "move": [move.x(), move.y()] if not move.is_pass() else None,
                                "confidence": round(move_stats.win_rate, 2),
                                "is_pass": move.is_pass(),
                            }
                        )
                else:
                    # Fallback: dùng best move và thêm legal moves
                    best_move = search_result.best_move
                    hints.append(
                        {
                            "move": [best_move.x(), best_move.y()] if not best_move.is_pass() else None,
                            "confidence": round(search_result.win_rate, 2) if hasattr(search_result, "win_rate") else 0.85,
                            "is_pass": best_move.is_pass(),
                        }
                    )

                    # Thêm legal moves nếu cần
                    if len(hints) < payload.top_k:
                        legal_moves = board.get_legal_moves()
                        for move in legal_moves[: payload.top_k - len(hints)]:
                            if move.is_pass():
                                continue
                            hints.append(
                                {
                                    "move": [move.x(), move.y()],
                                    "confidence": 0.5,
                                    "is_pass": False,
                                }
                            )

            except Exception as e:
                logger.error(f"Error generating hints: {e}", exc_info=True)
                # Fallback to simple heuristic
                hints = self._generate_fallback_hints(match, payload.top_k)
        else:
            # Fallback nếu không có gogame_py
            hints = self._generate_fallback_hints(match, payload.top_k)

        # Sort by confidence
        hints.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        hints = hints[: payload.top_k]

        collection = self.mongo_db.get_collection("premium_reports")
        await collection.insert_one(
            {
                "_id": request.id,
                "match_id": match.id,
                "feature": "hint",
                "summary": f"{len(hints)} đề xuất nước đi",
                "details": {"moves": hints},
                "coins_spent": self.HINT_COST,
                "created_at": datetime.now(tz=timezone.utc),
            }
        )

        return {"request_id": request.id, "hints": hints}

    def _generate_fallback_hints(self, match: match_model.Match, top_k: int) -> List[Dict[str, Any]]:
        """Fallback hints nếu không có gogame_py."""
        board_size = match.board_size
        # Star points và center
        star_points = [
            [3, 3],
            [3, board_size // 2],
            [3, board_size - 4],
            [board_size // 2, 3],
            [board_size // 2, board_size // 2],
            [board_size // 2, board_size - 4],
            [board_size - 4, 3],
            [board_size - 4, board_size // 2],
            [board_size - 4, board_size - 4],
        ]
        return [
            {"move": move, "confidence": 0.4 - i * 0.05, "is_pass": False}
            for i, move in enumerate(star_points[:top_k])
        ]

    async def create_analysis(
        self, user: user_model.User, match: match_model.Match, match_service: MatchService
    ) -> Dict[str, Any]:
        """Phân tích vị trí hiện tại sử dụng Value Network hoặc heuristic evaluation."""
        self._ensure_balance(user, self.ANALYSIS_COST)
        self.coin_service.add_transaction(user, -self.ANALYSIS_COST, "spend", source="premium_analysis")

        request = self._create_request(user, match, feature="analysis", cost=self.ANALYSIS_COST, status_value="completed")

        analysis_details: Dict[str, Any] = {}

        if go and self.ai_player:
            try:
                # Load board state
                board = await match_service._get_or_create_board(match)
                current_player = board.current_player()

                # Lấy Zobrist hash để dùng làm cache key
                board_hash = board.zobrist_hash()

                # Kiểm tra cache trước
                cache_key = (board_hash, current_player == go.Color.Black)
                cached_result = self.eval_cache.get(cache_key)

                if cached_result is not None:
                    logger.debug(f"Cache hit for analysis: hash={board_hash}")
                    evaluation = cached_result
                else:
                    # Cache miss, evaluate position
                    logger.debug(f"Cache miss for analysis: hash={board_hash}")
                    loop = asyncio.get_event_loop()
                    minimax_config = go.MinimaxConfig()
                    minimax_config.max_depth = 3
                    minimax_config.use_alpha_beta = True
                    minimax_engine = go.MinimaxEngine(minimax_config)

                    # Evaluate position bằng cách search
                    search_result = await loop.run_in_executor(
                        None,
                        lambda: minimax_engine.search(board, current_player),
                    )

                    evaluation = search_result.evaluation
                    # Cache kết quả
                    self.eval_cache.set(cache_key, evaluation)

                # Tính win probability từ evaluation score
                # Evaluation thường trong range [-100, 100] cho 9x9
                win_probability = max(0.0, min(1.0, (evaluation + 100) / 200))

                # Territory analysis
                territory_black = board.get_prisoners(go.Color.Black)
                territory_white = board.get_prisoners(go.Color.White)

                # Game phase estimation
                move_count = len((await self.mongo_db.get_collection("games").find_one({"match_id": match.id}) or {}).get("moves", []))
                board_size_sq = match.board_size * match.board_size
                if move_count < board_size_sq * 0.3:
                    phase = "opening"
                elif move_count < board_size_sq * 0.7:
                    phase = "middle"
                else:
                    phase = "endgame"

                analysis_details = {
                    "win_probability": round(win_probability, 3),
                    "evaluation_score": evaluation,
                    "territory_black": territory_black,
                    "territory_white": territory_white,
                    "current_player": "Black" if current_player == go.Color.Black else "White",
                    "game_phase": phase,
                    "move_count": move_count,
                    "recommendation": "advantage" if win_probability > 0.6 else "disadvantage" if win_probability < 0.4 else "balanced",
                }

            except Exception as e:
                logger.error(f"Error generating analysis: {e}", exc_info=True)
                analysis_details = {
                    "error": "Analysis generation failed",
                    "fallback": True,
                }
        else:
            # Fallback analysis
            game_doc = await self.mongo_db.get_collection("games").find_one({"match_id": match.id})
            move_count = len((game_doc or {}).get("moves", []))
            analysis_details = {
                "win_probability": 0.5,
                "evaluation_score": 0,
                "territory_black": (game_doc or {}).get("prisoners_black", 0),
                "territory_white": (game_doc or {}).get("prisoners_white", 0),
                "current_player": (game_doc or {}).get("current_player", "B"),
                "game_phase": "middle",
                "move_count": move_count,
                "recommendation": "balanced",
                "fallback": True,
            }

        collection = self.mongo_db.get_collection("premium_reports")
        await collection.insert_one(
            {
                "_id": request.id,
                "match_id": match.id,
                "feature": "analysis",
                "summary": f"Win probability: {analysis_details.get('win_probability', 0.5):.1%}",
                "details": analysis_details,
                "coins_spent": self.ANALYSIS_COST,
                "created_at": datetime.now(tz=timezone.utc),
            }
        )

        return {"request_id": request.id, "analysis": analysis_details}

    async def create_review(
        self, user: user_model.User, match: match_model.Match, match_service: MatchService
    ) -> Dict[str, Any]:
        """Tạo game review với phân tích toàn bộ ván đấu và mistakes detection."""
        self._ensure_balance(user, self.REVIEW_COST)
        self.coin_service.add_transaction(user, -self.REVIEW_COST, "spend", source="premium_review")
        request = self._create_request(user, match, feature="review", cost=self.REVIEW_COST, status_value="completed")

        review_details: Dict[str, Any] = {
            "mistakes": [],
            "key_moments": [],
            "statistics": {},
        }

        if go:
            try:
                # Load game history
                collection = self.mongo_db.get_collection("games")
                game_doc = await collection.find_one({"match_id": match.id})
                moves = (game_doc or {}).get("moves", [])

                if moves:
                    # Replay game và phân tích từng move
                    board = go.Board(match.board_size)
                    loop = asyncio.get_event_loop()
                    minimax_config = go.MinimaxConfig()
                    minimax_config.max_depth = 2
                    minimax_config.use_alpha_beta = True

                    mistakes = []
                    key_moments = []

                    for i, move_doc in enumerate(moves):
                        color = go.Color.Black if move_doc["color"] == "B" else go.Color.White
                        if move_doc.get("position"):
                            x, y = move_doc["position"]
                            move = go.Move(x, y, color)
                        else:
                            move = go.Move.Pass(color)

                        # Evaluate position trước move (với cache)
                        board_hash_before = board.zobrist_hash()
                        cache_key_before = (board_hash_before, color == go.Color.Black)
                        eval_before = self.eval_cache.get(cache_key_before)

                        if eval_before is None:
                            minimax_engine = go.MinimaxEngine(minimax_config)
                            search_before = await loop.run_in_executor(
                                None,
                                lambda: minimax_engine.search(board, color),
                            )
                            eval_before = search_before.evaluation
                            self.eval_cache.set(cache_key_before, eval_before)

                        # Apply move
                        if board.is_legal_move(move):
                            board.make_move(move)

                            # Evaluate position sau move (với cache)
                            board_hash_after = board.zobrist_hash()
                            cache_key_after = (board_hash_after, color == go.Color.Black)
                            eval_after = self.eval_cache.get(cache_key_after)

                            if eval_after is None:
                                minimax_engine_after = go.MinimaxEngine(minimax_config)
                                search_after = await loop.run_in_executor(
                                    None,
                                    lambda: minimax_engine_after.search(board, color),
                                )
                                eval_after = search_after.evaluation
                                self.eval_cache.set(cache_key_after, eval_after)

                            # Detect mistakes (evaluation giảm đáng kể)
                            eval_delta = eval_after - eval_before
                            if eval_delta < -10:  # Mistake threshold
                                mistakes.append(
                                    {
                                        "move_number": move_doc["number"],
                                        "color": move_doc["color"],
                                        "position": move_doc.get("position"),
                                        "eval_delta": round(eval_delta, 2),
                                        "severity": "major" if eval_delta < -20 else "minor",
                                    }
                                )

                            # Key moments (evaluation thay đổi lớn)
                            if abs(eval_delta) > 15:
                                key_moments.append(
                                    {
                                        "move_number": move_doc["number"],
                                        "color": move_doc["color"],
                                        "position": move_doc.get("position"),
                                        "eval_delta": round(eval_delta, 2),
                                        "type": "advantage_gain" if eval_delta > 0 else "advantage_loss",
                                    }
                                )

                    review_details["mistakes"] = mistakes[:10]  # Top 10 mistakes
                    review_details["key_moments"] = key_moments[:5]  # Top 5 key moments
                    review_details["statistics"] = {
                        "total_moves": len(moves),
                        "mistakes_count": len(mistakes),
                        "key_moments_count": len(key_moments),
                        "black_mistakes": len([m for m in mistakes if m["color"] == "B"]),
                        "white_mistakes": len([m for m in mistakes if m["color"] == "W"]),
                    }

            except Exception as e:
                logger.error(f"Error generating review: {e}", exc_info=True)
                review_details["error"] = "Review generation failed"
                review_details["fallback"] = True
        else:
            review_details["fallback"] = True
            review_details["statistics"] = {
                "total_moves": 0,
                "mistakes_count": 0,
                "key_moments_count": 0,
            }

        collection = self.mongo_db.get_collection("premium_reports")
        await collection.insert_one(
            {
                "_id": request.id,
                "match_id": match.id,
                "feature": "review",
                "summary": f"Review: {review_details.get('statistics', {}).get('mistakes_count', 0)} mistakes found",
                "details": review_details,
                "coins_spent": self.REVIEW_COST,
                "created_at": datetime.now(tz=timezone.utc),
            }
        )

        return {"request_id": request.id, "review": review_details}

    async def get_request(self, request_id: UUID) -> Dict[str, Any] | None:
        collection = self.mongo_db.get_collection("premium_reports")
        document = await collection.find_one({"_id": str(request_id)})
        if not document:
            return None
        return {
            "id": document["_id"],
            "feature": document["feature"],
            "summary": document.get("summary"),
            "details": document.get("details"),
            "coins_spent": document.get("coins_spent"),
        }

