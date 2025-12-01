"""Router cho matchmaking (ghép người chơi online)."""

from typing import Annotated
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_current_user, get_matchmaking_service, get_db, get_match_service
from ..models.sql import user as user_models
from ..models.sql import match as match_model
from ..schemas import matchmaking as matchmaking_schema
from ..schemas import matches as match_schema
from ..services.matchmaking_service import MatchmakingService
from ..services.match_service import MatchService
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/queue/join/test")
def test_join_queue():
    """Endpoint test đơn giản KHÔNG CẦN AUTH để kiểm tra POST request có đến được không."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("🧪 [TEST] POST /matchmaking/queue/join/test - REQUEST RECEIVED!")
    print("🧪 [TEST] POST /matchmaking/queue/join/test - REQUEST RECEIVED! (print)")
    return {"message": "Test endpoint works!", "status": "ok"}


@router.post("/queue/join", response_model=matchmaking_schema.QueueStatusResponse)
def join_queue(
    payload: matchmaking_schema.JoinQueueRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    matchmaking_service: Annotated[MatchmakingService, Depends(get_matchmaking_service)],
    db: Annotated[Session, Depends(get_db)],
):
    """Tham gia matchmaking queue.
    
    Tự động resign các matches đang active trước khi join queue.
    """
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    # LOG NGAY ĐẦU TIÊN - để biết request đã đến endpoint chưa
    logger.info(f"🚀 [JOIN_QUEUE] ===== REQUEST RECEIVED ===== User {current_user.id}, board_size={payload.board_size}")
    print(f"🚀 [JOIN_QUEUE] ===== REQUEST RECEIVED ===== User {current_user.id}, board_size={payload.board_size}")
    
    # LOG NGAY SAU KHI VÀO FUNCTION
    logger.info(f"✅ [JOIN_QUEUE] Function body started")
    print(f"✅ [JOIN_QUEUE] Function body started")
    
    start_time = time.time()
    logger.info(f"✅ [JOIN_QUEUE] Start time recorded: {start_time}")
    print(f"✅ [JOIN_QUEUE] Start time recorded: {start_time}")
    
    try:
        logger.info(f"✅ [JOIN_QUEUE] User {current_user.id} attempting to join queue for board size {payload.board_size}")
        print(f"✅ [JOIN_QUEUE] User {current_user.id} attempting to join queue for board size {payload.board_size}")
        
        # Check if user has active matches (nhanh, không block) - DÙNG DB TRỰC TIẾP
        # Chỉ check số lượng, không cần load toàn bộ matches
        check_start = time.time()
        logger.info(f"🔍 [JOIN_QUEUE] Starting active matches query...")
        print(f"🔍 [JOIN_QUEUE] Starting active matches query...")
        try:
            active_matches_count = (
                db.query(match_model.Match) # Use direct db
                .filter(
                    (
                        (match_model.Match.black_player_id == str(current_user.id))
                        | (match_model.Match.white_player_id == str(current_user.id))
                    )
                    & (match_model.Match.result.is_(None))
                    & (match_model.Match.finished_at.is_(None))
                )
                .count()
            )
            check_time = time.time() - check_start
            logger.info(f"⏱️ [JOIN_QUEUE] Active matches check took {check_time:.3f}s, count: {active_matches_count}")
            print(f"⏱️ [JOIN_QUEUE] Active matches check took {check_time:.3f}s, count: {active_matches_count}")
        except Exception as e:
            check_time = time.time() - check_start
            logger.error(f"❌ [JOIN_QUEUE] Error querying active matches after {check_time:.3f}s: {e}", exc_info=True)
            print(f"❌ [JOIN_QUEUE] Error querying active matches: {e}")
            # Continue without auto-resign if query fails
            active_matches_count = 0
        
        if active_matches_count > 0:
            logger.info(f"🔄 [JOIN_QUEUE] User {current_user.id} has {active_matches_count} active matches, auto-resigning...")
            # Auto-resign active matches - CHỈ resign match đầu tiên và skip ELO update để không block
            resign_start = time.time()
            try:
                # Get match service without MongoDB dependency for auto-resign
                from ..dependencies import get_match_service_no_mongo
                match_service = get_match_service_no_mongo(db) # Use the new dependency
                active_matches = match_service.get_active_matches_for_user(current_user.id)
                if active_matches:
                    # CHỈ resign match đầu tiên để nhanh
                    match = active_matches[0]
                    try:
                        # Resign match nhưng SKIP ELO update để không block
                        match.result = "W+R" if str(current_user.id) == match.black_player_id else "B+R"
                        match.finished_at = datetime.now(timezone.utc)
                        db.commit() # Use direct db commit
                        db.refresh(match) # Use direct db refresh
                        resign_time = time.time() - resign_start
                        logger.info(f"✅ [JOIN_QUEUE] Auto-resigned match {match.id} in {resign_time:.3f}s (skipped ELO update for speed)")
                        
                        # ELO update sẽ được thực hiện sau (background task hoặc khi match được query lại)
                        # Không block join queue request
                    except Exception as e:
                        logger.warning(f"❌ [JOIN_QUEUE] Error auto-resigning match {match.id}: {e}")
                
                if active_matches_count > 1:
                    logger.info(f"ℹ️ [JOIN_QUEUE] User {current_user.id} has {active_matches_count} active matches, only resigned first one for speed")
            except Exception as e:
                logger.warning(f"❌ [JOIN_QUEUE] Error auto-resigning matches: {e}")
                # Không fail request nếu auto-resign lỗi - user vẫn có thể join queue
        
        # Join queue (nhanh, chỉ thêm vào queue)
        join_start = time.time()
        logger.info(f"🔄 [JOIN_QUEUE] Calling matchmaking_service.join_queue()...")
        print(f"🔄 [JOIN_QUEUE] Calling matchmaking_service.join_queue()...")
        try:
            success = matchmaking_service.join_queue(
                user_id=str(current_user.id),
                elo_rating=current_user.elo_rating,
                board_size=payload.board_size
            )
            join_time = time.time() - join_start
            logger.info(f"⏱️ [JOIN_QUEUE] join_queue() took {join_time:.3f}s, success: {success}")
            print(f"⏱️ [JOIN_QUEUE] join_queue() took {join_time:.3f}s, success: {success}")
        except Exception as e:
            join_time = time.time() - join_start
            logger.error(f"❌ [JOIN_QUEUE] Error in join_queue() after {join_time:.3f}s: {e}", exc_info=True)
            print(f"❌ [JOIN_QUEUE] Error in join_queue(): {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi tham gia queue: {str(e)}"
            )
        
        if not success:
            logger.warning(f"⚠️ [JOIN_QUEUE] User {current_user.id} already in queue")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Bạn đã có trong queue rồi"
            )
        
        # Return queue status (nhanh, chỉ đọc từ memory)
        status_start = time.time()
        logger.info(f"📊 [JOIN_QUEUE] Getting queue status...")
        print(f"📊 [JOIN_QUEUE] Getting queue status...")
        try:
            status_info = matchmaking_service.get_queue_status(str(current_user.id))
            status_time = time.time() - status_start
            logger.info(f"⏱️ [JOIN_QUEUE] get_queue_status() took {status_time:.3f}s")
            print(f"⏱️ [JOIN_QUEUE] get_queue_status() took {status_time:.3f}s")
            
            if not status_info:
                logger.error(f"❌ [JOIN_QUEUE] Failed to get queue status for user {current_user.id}")
                print(f"❌ [JOIN_QUEUE] Failed to get queue status")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Không thể lấy trạng thái queue"
                )
            
            total_time = time.time() - start_time
            logger.info(f"✅ [JOIN_QUEUE] User {current_user.id} successfully joined queue in {total_time:.3f}s: {status_info}")
            print(f"✅ [JOIN_QUEUE] Successfully joined queue in {total_time:.3f}s")
            
            logger.info(f"📦 [JOIN_QUEUE] Creating response object from status_info: {status_info}")
            print(f"📦 [JOIN_QUEUE] Creating response object...")
            try:
                response = matchmaking_schema.QueueStatusResponse(**status_info)
                logger.info(f"✅ [JOIN_QUEUE] Response object created successfully: {response}")
                print(f"✅ [JOIN_QUEUE] Response object created successfully")
            except Exception as e:
                logger.error(f"❌ [JOIN_QUEUE] Error creating response object: {e}", exc_info=True)
                print(f"❌ [JOIN_QUEUE] Error creating response object: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Lỗi khi tạo response: {str(e)}"
                )
            
            logger.info(f"📤 [JOIN_QUEUE] Returning response to client")
            print(f"📤 [JOIN_QUEUE] Returning response to client")
            return response
        except HTTPException:
            raise
        except Exception as e:
            status_time = time.time() - status_start
            logger.error(f"❌ [JOIN_QUEUE] Error getting queue status after {status_time:.3f}s: {e}", exc_info=True)
            print(f"❌ [JOIN_QUEUE] Error getting queue status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi lấy trạng thái queue: {str(e)}"
            )
    except HTTPException as e:
        logger.error(f"❌ [JOIN_QUEUE] HTTPException: {e.detail}")
        print(f"❌ [JOIN_QUEUE] HTTPException: {e.detail}")
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ [JOIN_QUEUE] Unexpected error joining queue after {total_time:.3f}s: {e}", exc_info=True)
        print(f"❌ [JOIN_QUEUE] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tham gia queue: {str(e)}"
        )
    finally:
        # Đảm bảo log được ghi ngay cả khi có lỗi
        total_time = time.time() - start_time
        logger.info(f"🏁 [JOIN_QUEUE] Endpoint execution completed in {total_time:.3f}s")
        print(f"🏁 [JOIN_QUEUE] Endpoint execution completed in {total_time:.3f}s")


@router.post("/queue/leave")
def leave_queue(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    matchmaking_service: Annotated[MatchmakingService, Depends(get_matchmaking_service)],
):
    """Rời khỏi matchmaking queue."""
    matchmaking_service.leave_queue(str(current_user.id))
    return {"message": "Đã rời khỏi queue"}


@router.get("/queue/status", response_model=matchmaking_schema.QueueStatusResponse)
def get_queue_status(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    matchmaking_service: Annotated[MatchmakingService, Depends(get_matchmaking_service)],
    db: Annotated[Session, Depends(get_db)],
):
    """Lấy trạng thái queue của user hiện tại.
    
    Nếu user không còn trong queue nhưng có match đang chờ ready, 
    vẫn trả về in_queue=True để frontend tiếp tục check match.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    status_info = matchmaking_service.get_queue_status(str(current_user.id))
    
    # Nếu không còn trong queue, check xem có match đang chờ ready không
    if not status_info:
        # Check match đang chờ ready
        match = matchmaking_service.get_match_for_user(str(current_user.id))
        if match and (not match.black_ready or not match.white_ready):
            # Có match nhưng chưa cả 2 ready - vẫn coi như "in queue" để frontend tiếp tục check
            logger.info(
                f"ℹ️ User {current_user.id} not in queue but has match {match.id} waiting for ready"
            )
            return matchmaking_schema.QueueStatusResponse(
                in_queue=True,  # Vẫn trả về True để frontend tiếp tục check match
                board_size=match.board_size,
                elo_rating=current_user.elo_rating,
                wait_time=0,
                queue_size=1,
                elo_range=0
            )
        return matchmaking_schema.QueueStatusResponse(in_queue=False)
    
    return matchmaking_schema.QueueStatusResponse(**status_info)


@router.get("/queue/match")
def check_match(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    matchmaking_service: Annotated[MatchmakingService, Depends(get_matchmaking_service)],
    db: Annotated[Session, Depends(get_db)],
):
    """Kiểm tra xem đã có match được tạo chưa.
    
    Returns:
        MatchResponse nếu đã có match, None nếu chưa có.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"🔍 [CHECK_MATCH] Checking match for user {current_user.id}")
        
        # Retry logic để đảm bảo tìm thấy match ngay sau khi được tạo
        match = None
        max_retries = 5  # Tăng số lần retry
        import time
        for attempt in range(max_retries):
            match = matchmaking_service.get_match_for_user(str(current_user.id))
            if match:
                logger.info(f"✅ [CHECK_MATCH] Found match on attempt {attempt + 1}")
                break
            if attempt < max_retries - 1:
                # Tăng delay mỗi lần retry: 100ms, 200ms, 300ms, 400ms
                delay = 0.1 * (attempt + 1)
                time.sleep(delay)
                logger.debug(f"🔄 [CHECK_MATCH] Retry {attempt + 1}/{max_retries} for user {current_user.id} (waited {delay}s)")
        
        if not match:
            logger.info(f"ℹ️ [CHECK_MATCH] No match found for user {current_user.id} after {max_retries} attempts")
            # Return empty dict instead of None để frontend có thể check
            return {"matched": False}
        
        logger.info(
            f"✅ [CHECK_MATCH] Found match {match.id} for user {current_user.id} "
            f"(room_code={match.room_code}, black={match.black_player_id}, white={match.white_player_id})"
        )
        
        # Convert match to MatchResponse
        from ..routers.matches import _to_match_response
        response = _to_match_response(match, db_session=db, current_user_id=str(current_user.id))
        logger.info(
            f"📤 [CHECK_MATCH] Returning match response "
            f"(id={response.id}, room_code={response.room_code}, "
            f"black_ready={response.black_ready}, white_ready={response.white_ready})"
        )
        # Wrap in dict với matched flag
        return {"matched": True, "match": response}
    except Exception as e:
        logger.error(f"❌ [CHECK_MATCH] Error checking match: {e}", exc_info=True)
        # Return empty dict instead of None để frontend có thể check
        return {"matched": False}


@router.get("/queue/stats", response_model=matchmaking_schema.QueueStatsResponse)
def get_queue_stats(
    matchmaking_service: Annotated[MatchmakingService, Depends(get_matchmaking_service)],
):
    """Lấy thống kê tổng quan của queue (không cần auth)."""
    stats = matchmaking_service.get_queue_stats()
    return matchmaking_schema.QueueStatsResponse(**stats)
