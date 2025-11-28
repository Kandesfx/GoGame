"""Background jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from ..database import get_mongo_client
from ..utils.evaluation_cache import get_evaluation_cache

logger = logging.getLogger(__name__)


async def run_premium_job(request_id: str) -> dict[str, Any]:
    """Placeholder xử lý phân tích chuyên sâu."""

    # TODO: trigger AI analysis thực sự
    return {"request_id": request_id, "status": "completed"}


async def cleanup_evaluation_cache(interval_seconds: int = 300) -> None:
    """Background task để cleanup expired cache entries định kỳ.
    
    Args:
        interval_seconds: Khoảng thời gian giữa các lần cleanup (default: 5 phút)
    """
    cache = get_evaluation_cache()
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            removed = cache.cleanup_expired()
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache entries")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}", exc_info=True)


async def export_match_sgf(match_id: str, moves: list[Dict[str, Any]], board_size: int) -> str:
    """Export match sang SGF format trong background.
    
    Args:
        match_id: Match ID
        moves: List of moves
        board_size: Board size
    
    Returns:
        SGF string
    """
    from ..utils.sgf import export_sgf

    try:
        sgf_content = export_sgf(moves, board_size)
        
        # Lưu SGF vào MongoDB hoặc file system
        mongo_client = get_mongo_client()
        db = mongo_client["gogame"]
        collection = db.get_collection("sgf_files")
        
        from datetime import datetime, timezone

        await collection.insert_one(
            {
                "match_id": match_id,
                "sgf_content": sgf_content,
                "created_at": datetime.now(timezone.utc),
            }
        )
        
        logger.info(f"Exported SGF for match {match_id}")
        return sgf_content
    except Exception as e:
        logger.error(f"Error exporting SGF for match {match_id}: {e}", exc_info=True)
        raise


async def update_user_statistics(interval_seconds: int = 3600) -> None:
    """Background task để update user statistics định kỳ.
    
    Args:
        interval_seconds: Khoảng thời gian giữa các lần update (default: 1 giờ)
    """
    from ..database import SessionLocal
    from ..models.sql import match as match_model
    from ..models.sql import user as user_model

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logger.info("Starting user statistics update...")
            
            db = SessionLocal()
            try:
                # Update Elo ratings, win rates, etc.
                users = db.query(user_model.User).all()
                for user in users:
                    # Count matches
                    matches = (
                        db.query(match_model.Match)
                        .filter(
                            ((match_model.Match.black_player_id == user.id) | (match_model.Match.white_player_id == user.id))
                            & (match_model.Match.result.isnot(None))
                        )
                        .all()
                    )
                    
                    # TODO: Calculate win rate, update Elo, etc.
                    # Hiện tại chỉ log
                    if matches:
                        logger.debug(f"User {user.id} has {len(matches)} completed matches")
                
                logger.info(f"Updated statistics for {len(users)} users")
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in statistics update task: {e}", exc_info=True)


async def process_ml_training_job(job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process ML training job trong background.
    
    Args:
        job_id: Training job ID
        config: Training configuration
    
    Returns:
        Job result
    """
    try:
        logger.info(f"Starting ML training job {job_id} with config: {config}")
        
        # TODO: Integrate với actual ML training pipeline
        # Hiện tại chỉ là placeholder
        await asyncio.sleep(1)  # Simulate work
        
        logger.info(f"Completed ML training job {job_id}")
        return {
            "job_id": job_id,
            "status": "completed",
            "model_id": None,  # TODO: Return actual model ID
        }
    except Exception as e:
        logger.error(f"Error in ML training job {job_id}: {e}", exc_info=True)
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }

