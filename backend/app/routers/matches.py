"""Router quản lý trận đấu."""

from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_current_user, get_match_service, get_premium_service, get_db
from sqlalchemy.orm import Session
from ..models.sql import user as user_models
from ..schemas import matches as match_schema
from ..services.match_service import MatchService
from ..services.premium_service import PremiumService

router = APIRouter()


def _to_match_response(match, db_session=None, current_user_id=None) -> match_schema.MatchResponse:
    # Lấy username từ relationship nếu có
    black_username = None
    white_username = None
    
    # Eager load relationships nếu có
    if hasattr(match, 'black_player') and match.black_player:
        black_username = match.black_player.username
    elif match.black_player_id and db_session:
        # Nếu relationship chưa được load, query từ database
        from ..models.sql import user as user_model
        black_user = db_session.get(user_model.User, match.black_player_id)
        if black_user:
            black_username = black_user.username
    
    if hasattr(match, 'white_player') and match.white_player:
        white_username = match.white_player.username
    elif match.white_player_id and db_session:
        from ..models.sql import user as user_model
        white_user = db_session.get(user_model.User, match.white_player_id)
        if white_user:
            white_username = white_user.username
    
    # Tính toán user_elo_change và user_color dựa trên current_user
    user_elo_change = None
    user_color = None
    
    # QUAN TRỌNG: user_color phải được tính dựa trên player_id, KHÔNG phụ thuộc vào elo_change
    # Vì elo_change chỉ có sau khi match kết thúc, nhưng user_color cần có ngay từ đầu
    if current_user_id:
        if match.black_player_id and str(match.black_player_id) == str(current_user_id):
            user_color = "B"
            # Chỉ set user_elo_change nếu match đã kết thúc (có elo_change)
            if match.black_elo_change is not None:
                user_elo_change = match.black_elo_change
        elif match.white_player_id and str(match.white_player_id) == str(current_user_id):
            user_color = "W"
            # Chỉ set user_elo_change nếu match đã kết thúc (có elo_change)
            if match.white_elo_change is not None:
                user_elo_change = match.white_elo_change
    
    return match_schema.MatchResponse(
        id=UUID(match.id),
        board_size=match.board_size,
        ai_level=match.ai_level,
        result=match.result,
        started_at=match.started_at,
        finished_at=match.finished_at,
        black_player_id=UUID(match.black_player_id) if match.black_player_id else None,
        white_player_id=UUID(match.white_player_id) if match.white_player_id else None,
        black_player_username=black_username,
        white_player_username=white_username,
        premium_analysis_id=match.premium_analysis_id,
        room_code=match.room_code,
        black_elo_change=match.black_elo_change,
        white_elo_change=match.white_elo_change,
        user_elo_change=user_elo_change,
        user_color=user_color,
        black_ready=getattr(match, 'black_ready', False),
        white_ready=getattr(match, 'white_ready', False),
        state=None,
    )


@router.post("/ai", response_model=match_schema.MatchResponse, status_code=status.HTTP_201_CREATED)
def create_ai_match(
    payload: match_schema.MatchCreateAIRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.create_ai_match(current_user, payload)
    return _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id))


@router.post("/pvp", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_pvp_match(
    payload: match_schema.MatchCreatePVPRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match, code = match_service.create_pvp_match(current_user, payload)
    return {"match": _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id)), "join_code": code}


@router.post("/pvp/join", response_model=match_schema.MatchResponse)
def join_pvp_match_by_code(
    payload: match_schema.JoinByCodeRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Join PvP match bằng mã bàn 6 ký tự."""
    try:
        match = match_service.join_pvp_match_by_code(payload.room_code, current_user)
        return _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/pvp/{match_id}/join", response_model=match_schema.MatchResponse)
def join_pvp_match(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Join PvP match bằng match_id (legacy endpoint)."""
    try:
        match = match_service.join_pvp_match(match_id, current_user)
        return _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/history", response_model=List[match_schema.MatchResponse])
def list_history(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of matches to return")] = 100,
    offset: Annotated[int, Query(ge=0, description="Number of matches to skip")] = 0,
):
    """Lấy lịch sử matches của user."""
    matches = match_service.list_user_matches(UUID(current_user.id), limit=limit, offset=offset)
    return [_to_match_response(match, db_session=match_service.db) for match in matches]


@router.get("/{match_id}", response_model=match_schema.MatchResponse)
async def get_match(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.get_match(match_id)
    # TODO: kiểm tra quyền truy cập
    response = _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id))
    # Load game state từ MongoDB
    state = await match_service.get_match_state(match)
    if state:
        response.state = match_schema.MatchState(
            size=match.board_size,
            to_move=state["current_player"],
            moves=[tuple(m["position"]) for m in state["moves"] if m.get("position")],
            board_position=state.get("board_position"),  # Board state hiện tại (có xử lý capture)
            prisoners_black=state.get("prisoners_black", 0),
            prisoners_white=state.get("prisoners_white", 0),
            black_time_remaining_seconds=state.get("black_time_remaining_seconds"),
            white_time_remaining_seconds=state.get("white_time_remaining_seconds"),
        )
    return response


@router.post("/{match_id}/move", response_model=dict)
async def submit_move(
    match_id: UUID,
    payload: match_schema.MoveRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        match = match_service.get_match(match_id)
        if current_user.id not in {match.black_player_id, match.white_player_id}:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
        
        result = await match_service.record_move(match, payload, current_user_id=str(current_user.id))
        return result
    except HTTPException:
        # Re-raise HTTPException để FastAPI handle đúng
        raise
    except ValueError as e:
        # ValueError từ match_service (như "Đối thủ đã rời khỏỏi trận đấu")
        logger.warning(f"⚠️ [MOVE] ValueError in submit_move: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Catch tất cả exceptions khác để tránh generator error
        logger.error(f"❌ [MOVE] Unexpected error in submit_move: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi thực hiện nước đi: {str(e)}"
        )


@router.post("/{match_id}/pass", response_model=dict)
async def pass_turn(
    match_id: UUID,
    payload: match_schema.PassRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.get_match(match_id)
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    result = await match_service.pass_turn(match, payload.move_number, payload.color)
    return result


@router.post("/{match_id}/undo", response_model=dict)
async def undo_move(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Hoàn tác nước đi cuối cùng của user."""
    match = match_service.get_match(match_id)
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    
    try:
        result = await match_service.undo_move(match, str(current_user.id))
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{match_id}/ready")
def set_match_ready(
    match_id: UUID,
    payload: match_schema.MatchReadyRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Set ready status cho match (chỉ cho PvP matches).
    
    Chỉ bắt đầu trận đấu khi cả 2 người chơi đều ready.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"🔍 [SET_READY] User {current_user.id} setting ready={payload.ready} for match {match_id}")
        match = match_service.get_match(match_id)
        if not match:
            logger.error(f"❌ [SET_READY] Match {match_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Match không tồn tại. Match ID: {match_id}"
            )
        
        # Chỉ cho phép PvP matches
        if match.ai_level is not None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ready status chỉ áp dụng cho PvP matches")
        
        # Kiểm tra user có phải là player trong match không
        user_is_black = str(match.black_player_id) == str(current_user.id)
        user_is_white = str(match.white_player_id) == str(current_user.id)
        
        if not user_is_black and not user_is_white:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bạn không phải là người chơi trong match này")
        
        # Set ready status
        if user_is_black:
            match.black_ready = payload.ready
            logger.info(f"✅ Black player {current_user.id} set ready={payload.ready} for match {match_id}")
        else:
            match.white_ready = payload.ready
            logger.info(f"✅ White player {current_user.id} set ready={payload.ready} for match {match_id}")
        
        match_service.db.commit()
        match_service.db.refresh(match)
        
        # Check if both players are ready
        both_ready = match.black_ready and match.white_ready
        logger.info(
            f"📊 Match {match_id} ready status: black={match.black_ready}, white={match.white_ready}, "
            f"both_ready={both_ready}, room_code={match.room_code}"
        )
        
        return {
            "ready": payload.ready,
            "black_ready": match.black_ready,
            "white_ready": match.white_ready,
            "both_ready": both_ready
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error setting ready status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi set ready status: {str(e)}"
        )


@router.delete("/{match_id}", status_code=status.HTTP_200_OK)
def cancel_match(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Hủy match khi chưa có người chơi thứ 2."""
    match = match_service.get_match(match_id)
    try:
        match_service.cancel_match(match, current_user)
        return {"message": "Match đã được hủy thành công", "cancelled": True}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{match_id}/resign", response_model=match_schema.MatchResponse)
def resign_match(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    match = match_service.get_match(match_id)
    match = match_service.resign_match(match, current_user)
    return _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id))


@router.get("/{match_id}/analysis", response_model=dict)
async def get_analysis(
    match_id: UUID,
    request_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)],
):
    report = await premium_service.get_request(request_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.get("/{match_id}/sgf", response_model=dict)
async def export_sgf(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Export match sang SGF format."""
    match = match_service.get_match(match_id)
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    
    try:
        sgf_content = await match_service.export_sgf(match)
        return {
            "match_id": str(match_id),
            "sgf_content": sgf_content,
            "sgf_id": match.sgf_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/import-sgf", response_model=match_schema.MatchResponse, status_code=status.HTTP_201_CREATED)
async def import_sgf(
    payload: match_schema.SGFImportRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Import game từ SGF format."""
    try:
        match = await match_service.import_sgf(current_user, payload.sgf_content)
        return _to_match_response(match, db_session=match_service.db, current_user_id=str(current_user.id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{match_id}/replay", response_model=dict)
async def get_replay(
    match_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """Lấy replay data cho match (moves theo thứ tự)."""
    match = match_service.get_match(match_id)
    # Allow access if user is in match or if match is finished (for viewing replays)
    if match.finished_at and current_user.id not in {match.black_player_id, match.white_player_id}:
        # Allow viewing finished matches
        pass
    elif current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not in match")
    
    try:
        replay_data = await match_service.get_replay(match)
        return replay_data
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

