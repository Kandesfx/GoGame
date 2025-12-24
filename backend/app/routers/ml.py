"""Router quản trị ML và phân tích vị trí."""

from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import (
    get_current_user,
    get_ml_service,
    get_ml_position_analysis_service_dep,
    get_match_service,
    get_coin_service,
)
from ..models.sql import user as user_models
from ..schemas import ml as ml_schema
from ..services.ml_service import MLService
from ..services.ml_position_analysis_service import MLPositionAnalysisService
from ..services.match_service import MatchService
from ..services.coin_service import CoinService

router = APIRouter()


def ensure_admin(current_user) -> None:
    # TODO: kiểm tra quyền admin (tạm thời cho phép tất cả)
    return None


# Admin endpoints
@router.post("/train", response_model=dict)
async def trigger_training(
    payload: ml_schema.TrainRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[MLService, Depends(get_ml_service)],
):
    ensure_admin(current_user)
    return await service.trigger_training(payload)


@router.get("/models", response_model=List[ml_schema.ModelVersion])
async def list_models(
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[MLService, Depends(get_ml_service)],
):
    ensure_admin(current_user)
    return await service.list_models()


@router.post("/models/{model_id}/promote", response_model=dict)
async def promote_model(
    model_id: UUID,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    service: Annotated[MLService, Depends(get_ml_service)],
):
    ensure_admin(current_user)
    return await service.promote_model(model_id)


# Analysis endpoints
@router.post("/analyze-position", response_model=ml_schema.PositionAnalysisResponse)
async def analyze_position(
    request: ml_schema.AnalyzePositionRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    analysis_service: Annotated[
        Optional[MLPositionAnalysisService], Depends(get_ml_position_analysis_service_dep)
    ],
    match_service: Annotated[MatchService, Depends(get_match_service)],
    coin_service: Annotated[CoinService, Depends(get_coin_service)],
):
    """
    Phân tích vị trí hiện tại bằng ML.
    
    Cost: 50 coins
    """
    ANALYSIS_COST = 50

    # Check coins
    if current_user.coins < ANALYSIS_COST:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Không đủ coins. Cần {ANALYSIS_COST} coins để phân tích.",
        )

    # Verify match belongs to user
    match = match_service.get_match(request.match_id)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Match not found"
        )

    if current_user.id not in [match.black_player_id, match.white_player_id]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not your match"
        )

    # Check if analysis service is available
    if not analysis_service or not analysis_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML analysis service is not available. Model may not be loaded.",
        )

    # Convert move_history format
    move_history = None
    if request.move_history:
        move_history = [tuple(move) for move in request.move_history]

    # Run analysis
    try:
        analysis = analysis_service.analyze_position(
            board_position=request.board_position,
            current_player=request.current_player,
            board_size=request.board_size,
            move_history=move_history,
        )

        # Deduct coins
        coin_service.add_transaction(
            current_user, -ANALYSIS_COST, "spend", source="ml_analysis"
        )

        return ml_schema.PositionAnalysisResponse(
            threats=ml_schema.ThreatAnalysis(**analysis["threats"]),
            attacks=ml_schema.AttackAnalysis(**analysis["attacks"]),
            evaluation=ml_schema.PositionEvaluation(**analysis["evaluation"]),
            intent=ml_schema.IntentAnalysis(**analysis["intent"]),
            best_move=ml_schema.BestMove(**analysis["best_move"])
            if analysis.get("best_move")
            else None,
            fallback=analysis.get("fallback", False),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post("/analyze-position-from-match", response_model=ml_schema.PositionAnalysisResponse)
async def analyze_position_from_match(
    request: ml_schema.AnalyzePositionFromMatchRequest,
    current_user: Annotated[user_models.User, Depends(get_current_user)],
    analysis_service: Annotated[
        Optional[MLPositionAnalysisService], Depends(get_ml_position_analysis_service_dep)
    ],
    match_service: Annotated[MatchService, Depends(get_match_service)],
    coin_service: Annotated[CoinService, Depends(get_coin_service)],
):
    """
    Phân tích vị trí từ match hiện tại (tự động lấy board state từ match).
    
    Cost: 50 coins
    """
    ANALYSIS_COST = 50

    # Check coins
    if current_user.coins < ANALYSIS_COST:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Không đủ coins. Cần {ANALYSIS_COST} coins để phân tích.",
        )

    # Get match
    match = match_service.get_match(request.match_id)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Match not found"
        )

    if current_user.id not in [match.black_player_id, match.white_player_id]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not your match"
        )

    # Check if analysis service is available
    if not analysis_service or not analysis_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML analysis service is not available. Model may not be loaded.",
        )

    # Get board state from MongoDB directly (không cần gogame_py)
    try:
        # Sử dụng mongo_db từ match_service
        collection = match_service.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": match.id})
        
        if not game_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Game state not found"
            )

        # Get board_position nếu có (đã được tính sẵn)
        board_position = game_doc.get("board_position", {})
        
        # Nếu không có board_position, tính từ moves
        if not board_position:
            moves = game_doc.get("moves", [])
            board_position = {}
            for move in moves:
                if move and move.get("position"):
                    x, y = move["position"]
                    color = move.get("color", "B")
                    board_position[f"{x},{y}"] = color
                    
                    # Xóa captured stones
                    captured = move.get("captured", [])
                    for cap in captured:
                        if isinstance(cap, list) and len(cap) == 2:
                            cap_x, cap_y = cap
                            cap_key = f"{cap_x},{cap_y}"
                            if cap_key in board_position:
                                del board_position[cap_key]
        else:
            # Đã có board_position, chỉ cần lấy moves để tính move_history
            moves = game_doc.get("moves", [])

        # QUAN TRỌNG: Xác định current_player là người đang YÊU CẦU phân tích (không phải người đang đến lượt)
        # Phân tích threat phải là cho quân của người đang yêu cầu phân tích
        if current_user.id == match.black_player_id:
            current_player = "B"  # Người yêu cầu là Black
        elif current_user.id == match.white_player_id:
            current_player = "W"  # Người yêu cầu là White
        else:
            # Fallback: dùng current_player từ game_doc
            current_player = game_doc.get("current_player", "B")
            if not current_player:
                current_player = "B" if len(moves) % 2 == 0 else "W"
        
        board_size = game_doc.get("board_size", match.board_size)

        # Get move history (last 4 moves)
        move_history = None
        if len(moves) >= 4:
            move_history = [
                tuple(move["position"]) for move in moves[-4:] if move.get("position")
            ]

        # Run analysis
        analysis = analysis_service.analyze_position(
            board_position=board_position,
            current_player=current_player,
            board_size=board_size,
            move_history=move_history,
        )

        # Deduct coins
        coin_service.add_transaction(
            current_user, -ANALYSIS_COST, "spend", source="ml_analysis"
        )

        return ml_schema.PositionAnalysisResponse(
            threats=ml_schema.ThreatAnalysis(**analysis["threats"]),
            attacks=ml_schema.AttackAnalysis(**analysis["attacks"]),
            evaluation=ml_schema.PositionEvaluation(**analysis["evaluation"]),
            intent=ml_schema.IntentAnalysis(**analysis["intent"]),
            best_move=ml_schema.BestMove(**analysis["best_move"])
            if analysis.get("best_move")
            else None,
            fallback=analysis.get("fallback", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )
