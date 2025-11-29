"""Service quản lý trận đấu."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

try:
    import gogame_py as go
    _GOGAME_PY_DIRECT = True
except ImportError:
    go = None  # type: ignore
    _GOGAME_PY_DIRECT = False
    logging.warning("gogame_py module not found. AI features will be disabled.")
    # Try subprocess wrapper
    try:
        from ..utils.ai_wrapper import call_ai_select_move as _wrapper_select_move
        _GOGAME_PY_WRAPPER = True
    except ImportError:
        _GOGAME_PY_WRAPPER = False
        logging.warning("AI wrapper not available either.")

from uuid import UUID, uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from ..config import Settings
from ..models.sql import match as match_model
from ..models.sql import user as user_model
from ..schemas import matches as match_schema

logger = logging.getLogger(__name__)


class MatchService:
    def __init__(self, db: Session, mongo_db: AsyncIOMotorDatabase, settings: Settings) -> None:
        self.db = db
        self.mongo_db = mongo_db
        self.settings = settings
        self.ai_player = go.AIPlayer() if go else None

    def _calculate_capture_fallback(self, board_position: dict, x: int, y: int, color: str, board_size: int) -> List[Tuple[int, int]]:
        """
        Tính captured stones trong fallback mode (không có gogame_py).
        Logic: Sau khi đặt quân, kiểm tra các nhóm đối phương xung quanh có bị hết khí không.
        """
        captured = []
        opponent_color = "W" if color == "B" else "B"
        move_key = f"{x},{y}"
        
        # Tạo board_position sau khi đặt quân
        board_after = {**board_position, move_key: color}
        
        # Kiểm tra 4 neighbors
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]
        
        visited_groups = set()
        
        for nx, ny in neighbors:
            # Kiểm tra bounds
            if nx < 0 or nx >= board_size or ny < 0 or ny >= board_size:
                continue
            
            neighbor_key = f"{nx},{ny}"
            neighbor_stone = board_after.get(neighbor_key)
            
            # Chỉ kiểm tra quân đối phương
            if neighbor_stone != opponent_color:
                continue
            
            # Nếu đã kiểm tra nhóm này rồi thì skip
            if neighbor_key in visited_groups:
                continue
            
            # Thu thập nhóm đối phương (BFS)
            group_stones = []
            group_liberties = set()
            frontier = [(nx, ny)]
            visited = {neighbor_key}
            
            while frontier:
                cx, cy = frontier.pop(0)
                group_stones.append((cx, cy))
                visited_groups.add(f"{cx},{cy}")
                
                # Kiểm tra 4 neighbors của quân trong nhóm
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    tx, ty = cx + dx, cy + dy
                    
                    # Kiểm tra bounds
                    if tx < 0 or tx >= board_size or ty < 0 or ty >= board_size:
                        continue
                    
                    neighbor_pos_key = f"{tx},{ty}"
                    neighbor_pos_stone = board_after.get(neighbor_pos_key)
                    
                    # Nếu là quân cùng màu (cùng nhóm) → thêm vào frontier
                    if neighbor_pos_stone == opponent_color and neighbor_pos_key not in visited:
                        visited.add(neighbor_pos_key)
                        frontier.append((tx, ty))
                    # Nếu là vị trí trống → đây là khí (liberty)
                    elif neighbor_pos_stone is None:
                        group_liberties.add((tx, ty))
            
            # Nếu nhóm không còn khí → bắt toàn bộ nhóm
            if not group_liberties:
                captured.extend(group_stones)
        
        return captured
    
    def _check_ko_rule_fallback(self, board_position: dict, x: int, y: int, color: str, 
                                 captured_stones: List[Tuple[int, int]], board_size: int, 
                                 ko_position: Optional[Tuple[int, int]]) -> bool:
        """
        Kiểm tra Ko rule trong fallback mode.
        Ko rule: Không được đặt quân tại vị trí mà nước đi trước đó vừa bắt được một quân đơn lẻ.
        
        Returns:
            True nếu vi phạm Ko rule (illegal), False nếu hợp lệ.
        """
        # Nếu không có ko_position → không vi phạm Ko
        if ko_position is None:
            return False
        
        # Kiểm tra xem nước đi có đặt tại vị trí Ko không
        if (x, y) == ko_position:
            return True  # Vi phạm Ko rule
        
        return False
    
    def _calculate_ko_position_fallback(self, board_position: dict, x: int, y: int, color: str,
                                        captured_stones: List[Tuple[int, int]], board_size: int) -> Optional[Tuple[int, int]]:
        """
        Tính ko_position sau khi đặt quân trong fallback mode.
        Ko position được set khi:
        - Capture đúng 1 quân đối phương
        - Nhóm quân mình (sau khi đặt và xóa captured stones) chỉ có 1 quân
        
        Returns:
            Tuple (x, y) của ko_position nếu thỏa điều kiện, None nếu không.
        """
        # Ko rule chỉ áp dụng khi capture đúng 1 quân
        if len(captured_stones) != 1:
            return None
        
        # QUAN TRỌNG: Xây dựng board_after với captured stones đã bị xóa
        # Đây là board state thực tế sau khi đặt quân và capture
        move_key = f"{x},{y}"
        board_after = {**board_position, move_key: color}
        
        # Xóa captured stones khỏi board_after (chúng đã bị bắt)
        for cx, cy in captured_stones:
            captured_key = f"{cx},{cy}"
            if captured_key in board_after:
                del board_after[captured_key]
        
        # Thu thập nhóm quân mình tại vị trí vừa đặt (sau khi đã xóa captured stones)
        group_stones = []
        frontier = [(x, y)]
        visited = {move_key}
        
        while frontier:
            cx, cy = frontier.pop(0)
            group_stones.append((cx, cy))
            
            # Kiểm tra 4 neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                tx, ty = cx + dx, cy + dy
                
                # Kiểm tra bounds
                if tx < 0 or tx >= board_size or ty < 0 or ty >= board_size:
                    continue
                
                neighbor_key = f"{tx},{ty}"
                neighbor_stone = board_after.get(neighbor_key)
                
                # Nếu là quân cùng màu (cùng nhóm) → thêm vào frontier
                if neighbor_stone == color and neighbor_key not in visited:
                    visited.add(neighbor_key)
                    frontier.append((tx, ty))
        
        # Nếu nhóm chỉ có 1 quân → set ko_position = vị trí quân bị bắt
        if len(group_stones) == 1:
            return captured_stones[0]  # Vị trí quân bị bắt
        
        return None
    
    def get_active_matches_for_user(self, user_id: UUID) -> List[match_model.Match]:
        """Lấy tất cả matches đang active (chưa kết thúc) của user."""
        return (
            self.db.query(match_model.Match)
            .filter(
                (
                    (match_model.Match.black_player_id == str(user_id))
                    | (match_model.Match.white_player_id == str(user_id))
                )
                & (match_model.Match.result.is_(None))
                & (match_model.Match.finished_at.is_(None))
            )
            .all()
        )

    def auto_resign_active_matches(self, user: user_model.User, exclude_match_id: Optional[UUID] = None) -> int:
        """Tự động resign tất cả matches đang active của user (trừ match được exclude).
        
        Returns:
            Số lượng matches đã resign.
        """
        active_matches = self.get_active_matches_for_user(UUID(user.id))
        resigned_count = 0
        
        for match in active_matches:
            # Skip match nếu được exclude
            if exclude_match_id and str(match.id) == str(exclude_match_id):
                continue
            
            try:
                # Auto-resign match này
                self.resign_match(match, user)
                resigned_count += 1
                logger.info(f"Auto-resigned match {match.id} for user {user.id} (user started new match)")
            except Exception as e:
                logger.error(f"Failed to auto-resign match {match.id}: {e}", exc_info=True)
        
        return resigned_count

    def create_ai_match(self, user: user_model.User, request: match_schema.MatchCreateAIRequest) -> match_model.Match:
        # Auto-resign các matches đang active của user
        resigned_count = self.auto_resign_active_matches(user)
        if resigned_count > 0:
            print(f"Auto-resigned {resigned_count} active match(es) for user {user.id} before creating new AI match")
        
        # Xác định player color - nếu user chọn white thì AI là black (đi trước)
        player_color = request.player_color
        print(f"🎨 [SERVICE] Creating AI match with player_color={player_color} for user {user.id}")
        
        if player_color == 'white':
            # User chọn quân trắng -> AI là quân đen (đi trước)
            black_player_id = None  # AI
            white_player_id = user.id  # User
            print(f"🎨 [SERVICE] User chose WHITE: black_player_id=None (AI), white_player_id={user.id} (User)")
        else:
            # User chọn quân đen (mặc định) -> User đi trước
            black_player_id = user.id  # User
            white_player_id = None  # AI
            print(f"🎨 [SERVICE] User chose BLACK: black_player_id={user.id} (User), white_player_id=None (AI)")
        
        match = match_model.Match(
            black_player_id=black_player_id,
            white_player_id=white_player_id,
            ai_level=request.level,
            board_size=request.board_size,
        )
        self.db.add(match)
        self.db.commit()
        self.db.refresh(match)
        
        return match

    def create_pvp_match(
        self, user: user_model.User, request: match_schema.MatchCreatePVPRequest
    ) -> Tuple[match_model.Match, str]:
        # Auto-resign các matches đang active của user (bao gồm cả PvP và AI)
        resigned_count = self.auto_resign_active_matches(user)
        if resigned_count > 0:
            logger.info(f"Auto-resigned {resigned_count} active match(es) for user {user.id} before creating new PvP match")
        
        # Tạo mã bàn 6 ký tự duy nhất
        import random
        import string
        max_attempts = 10
        room_code = None
        for _ in range(max_attempts):
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            # Kiểm tra mã đã tồn tại chưa - chỉ check các match đang active (chưa kết thúc và chưa đủ người)
            existing = self.db.query(match_model.Match).filter(
                match_model.Match.room_code == code,
                match_model.Match.white_player_id.is_(None),  # Chưa đủ người
                match_model.Match.finished_at.is_(None),  # Chưa kết thúc
                match_model.Match.result.is_(None)  # Chưa có kết quả
            ).first()
            if not existing:
                room_code = code
                break
        
        if not room_code:
            # Fallback nếu không tạo được mã duy nhất
            room_code = uuid4().hex[:6].upper()
            logger.warning(f"Could not generate unique room code, using fallback: {room_code}")
        
        # Khởi tạo time control cho PvP match
        time_control_seconds = request.time_control_minutes * 60  # Chuyển phút sang giây
        
        match = match_model.Match(
            black_player_id=user.id,
            board_size=request.board_size,
            room_code=room_code,
            time_control_minutes=request.time_control_minutes,
            black_time_remaining_seconds=time_control_seconds,  # Black bắt đầu với đầy đủ thời gian
            white_time_remaining_seconds=None,  # White chưa join nên chưa có thời gian
            last_move_at=datetime.now(timezone.utc),  # Bắt đầu đếm thời gian từ khi tạo match
        )
        self.db.add(match)
        self.db.commit()
        self.db.refresh(match)

        return match, room_code

    def join_pvp_match_by_code(self, room_code: str, user: user_model.User) -> match_model.Match:
        """Join PvP match bằng mã bàn."""
        match = self.db.query(match_model.Match).filter(
            match_model.Match.room_code == room_code.upper()
        ).first()
        
        if not match:
            raise ValueError("Mã bàn không tồn tại")
        
        # Kiểm tra match đã kết thúc chưa
        if match.finished_at is not None or match.result is not None:
            raise ValueError("Bàn đấu đã kết thúc")
        
        if match.white_player_id:
            raise ValueError("Bàn đã có đủ người chơi")
        
        if match.black_player_id == str(user.id):
            raise ValueError("Bạn không thể tham gia bàn của chính mình")
        
        # Auto-resign các matches đang active của user (trừ match đang join)
        resigned_count = self.auto_resign_active_matches(user, exclude_match_id=UUID(match.id))
        if resigned_count > 0:
            logger.info(f"Auto-resigned {resigned_count} active match(es) for user {user.id} before joining PvP match {match.id}")
        
        match.white_player_id = user.id
        
        # Khởi tạo thời gian cho White player khi join
        if match.time_control_minutes and match.white_time_remaining_seconds is None:
            match.white_time_remaining_seconds = match.time_control_minutes * 60
            # Cập nhật last_move_at để bắt đầu đếm thời gian cho White
            if match.last_move_at is None:
                match.last_move_at = datetime.now(timezone.utc)
        
        self.db.commit()
        self.db.refresh(match)
        
        logger.info(f"User {user.id} joined PvP match {match.id} with room code {room_code}")
        return match

    def join_pvp_match(self, match_id: UUID, user: user_model.User) -> match_model.Match:
        match = self.get_match(match_id)
        if match.white_player_id:
            raise ValueError("Match đã có đủ người chơi")
        
        # Auto-resign các matches đang active của user (trừ match đang join)
        resigned_count = self.auto_resign_active_matches(user, exclude_match_id=match_id)
        if resigned_count > 0:
            logger.info(f"Auto-resigned {resigned_count} active match(es) for user {user.id} before joining PvP match {match_id}")
        
        # Nếu match đã có black_player và black_player tạo match mới → auto-resign match này
        if match.black_player_id and match.black_player_id != str(user.id):
            # Check nếu black_player có match active khác (đã tạo match mới)
            black_player_active = self.get_active_matches_for_user(UUID(match.black_player_id))
            black_player_has_other_match = any(str(m.id) != str(match_id) for m in black_player_active)
            if black_player_has_other_match:
                # Black player đã tạo match mới → auto-resign match này (black player thua)
                logger.info(f"Black player {match.black_player_id} has other active match, auto-resigning match {match_id}")
                self.resign_match(match, self.db.get(user_model.User, match.black_player_id))
                raise ValueError("Đối thủ đã rời khỏi trận đấu. Trận đấu đã kết thúc.")
        
        match.white_player_id = user.id
        self.db.commit()
        self.db.refresh(match)
        return match

    def get_match(self, match_id: UUID) -> match_model.Match:
        """Lấy match từ database.
        
        Raises:
            ValueError: Nếu match không tồn tại
        """
        import logging
        logger = logging.getLogger(__name__)
        
        match_id_str = str(match_id)
        logger.debug(f"🔍 [GET_MATCH] Looking for match: {match_id_str}")
        
        match = self.db.get(match_model.Match, match_id_str)
        if not match:
            logger.error(f"❌ [GET_MATCH] Match {match_id_str} not found in database")
            raise ValueError(f"Match không tồn tại. Match ID: {match_id_str}")
        
        logger.debug(f"✅ [GET_MATCH] Found match {match_id_str} (black={match.black_player_id}, white={match.white_player_id})")
        return match

    async def get_match_state(self, match: match_model.Match) -> dict | None:
        """Lấy game state từ MongoDB và trả về board state hiện tại."""
        collection = self.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": match.id})
        
        # Nếu chưa có game_doc, tạo mới
        if not game_doc:
            # Khởi tạo game document
            await collection.insert_one({
                "match_id": match.id,
                "moves": [],
                "board_position": {},
                "current_player": "B",
                "prisoners_black": 0,
                "prisoners_white": 0,
            })
            
            # Nếu AI đi trước (user chọn white trong AI match), AI cần đánh nước đầu tiên
            if match.ai_level and match.white_player_id and not match.black_player_id:
                print(f"🤖 AI goes first for match {match.id} (user chose white)")
                logger.info(f"AI goes first for match {match.id} (user chose white)")
                
                # Thử dùng gogame_py trực tiếp
                if go:
                    board = go.Board(match.board_size)
                    ai_result = await self._make_ai_move(match, board)
                    if ai_result:
                        print(f"🤖 AI first move completed: {ai_result}")
                    # Reload game_doc sau khi AI đánh
                    game_doc = await collection.find_one({"match_id": match.id})
                # Fallback: dùng wrapper nếu gogame_py không available
                elif _GOGAME_PY_WRAPPER:
                    print(f"🤖 Using wrapper for AI first move")
                    ai_result = await self._make_ai_move_wrapper(match)
                    if ai_result:
                        print(f"🤖 AI first move (wrapper) completed: {ai_result}")
                    game_doc = await collection.find_one({"match_id": match.id})
                else:
                    print(f"❌ Cannot make AI first move - no AI module available")
                    logger.warning(f"gogame_py not available, cannot make AI first move for match {match.id}")
            
            if not game_doc:
                return None
        
        moves = game_doc.get("moves", [])
        current_player = game_doc.get("current_player", "B")
        prisoners_black = game_doc.get("prisoners_black", 0)
        prisoners_white = game_doc.get("prisoners_white", 0)
        
        # Ưu tiên sử dụng board_position từ MongoDB (đã được cập nhật sau mỗi move)
        board_position = game_doc.get("board_position")
        
        # Nếu không có board_position trong MongoDB, rebuild từ moves
        if not board_position and go:
            try:
                logger.info(f"Rebuilding board_position from moves for match {match.id}")
                board = go.Board(match.board_size)
                # Replay tất cả moves để có board state hiện tại
                # board.make_move() sẽ tự động xử lý capture
                for move_doc in moves:
                    if move_doc.get("position"):
                        x, y = move_doc["position"]
                        color = go.Color.Black if move_doc["color"] == "B" else go.Color.White
                        move = go.Move(x, y, color)
                        if board.is_legal_move(move):
                            board.make_move(move)  # Tự động xử lý capture
                        else:
                            logger.warning(f"⚠️ Illegal move in history: {move_doc} - skipping")
                
                # Xây dựng board position từ board state (sau khi đã replay tất cả moves)
                board_position = {}
                for x in range(match.board_size):
                    for y in range(match.board_size):
                        stone = board.at(x, y)
                        if stone == go.Stone.Black:
                            board_position[f"{x},{y}"] = "B"
                        elif stone == go.Stone.White:
                            board_position[f"{x},{y}"] = "W"
                
                # Cập nhật prisoners từ board state (chính xác hơn)
                prisoners_black = board.get_prisoners(go.Color.Black)
                prisoners_white = board.get_prisoners(go.Color.White)
                
                # Lưu lại vào MongoDB để lần sau không cần rebuild
                await collection.update_one(
                    {"match_id": match.id},
                    {
                        "$set": {
                            "board_position": board_position,
                    "prisoners_black": prisoners_black,
                    "prisoners_white": prisoners_white,
                        }
                    }
                )
                logger.info(f"Rebuilt board_position: {len(board_position)} stones, prisoners: B={prisoners_black}, W={prisoners_white}")
            except Exception as e:
                logger.error(f"Error rebuilding board state: {e}", exc_info=True)
                board_position = None
        
        # Tính thời gian còn lại cho mỗi người chơi (chỉ cho PvP matches với time control)
        black_time_remaining = None
        white_time_remaining = None
        
        if match.time_control_minutes and match.last_move_at and not match.ai_level:
            now = datetime.now(timezone.utc)
            elapsed_seconds = int((now - match.last_move_at).total_seconds())
            
            # Tính thời gian còn lại cho người chơi hiện tại (đang đến lượt)
            if current_player == "B" and match.black_time_remaining_seconds is not None:
                black_time_remaining = max(0, match.black_time_remaining_seconds - elapsed_seconds)
                white_time_remaining = match.white_time_remaining_seconds  # White chưa đến lượt nên giữ nguyên
            elif current_player == "W" and match.white_time_remaining_seconds is not None:
                white_time_remaining = max(0, match.white_time_remaining_seconds - elapsed_seconds)
                black_time_remaining = match.black_time_remaining_seconds  # Black chưa đến lượt nên giữ nguyên
        else:
            # Nếu không có thời gian được khởi tạo, sử dụng giá trị từ database
            black_time_remaining = match.black_time_remaining_seconds
            white_time_remaining = match.white_time_remaining_seconds
        
        return {
            "moves": moves,
            "current_player": current_player,
            "prisoners_black": prisoners_black,
            "prisoners_white": prisoners_white,
            "board_position": board_position,  # Board state hiện tại (từ MongoDB hoặc rebuilt)
            "black_time_remaining_seconds": black_time_remaining,
            "white_time_remaining_seconds": white_time_remaining,
        }

    def list_user_matches(self, user_id: UUID, limit: int = 20, offset: int = 0) -> List[match_model.Match]:
        return (
            self.db.query(match_model.Match)
            .filter(
                (match_model.Match.black_player_id == str(user_id))
                | (match_model.Match.white_player_id == str(user_id))
            )
            .order_by(match_model.Match.started_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def check_opponent_disconnected(self, match: match_model.Match, current_user_id: str) -> bool:
        """Kiểm tra nếu đối thủ đã disconnect (có match active khác).
        
        Returns:
            True nếu đối thủ đã disconnect, False nếu không.
        """
        # Xác định đối thủ
        if match.black_player_id == current_user_id:
            opponent_id = match.white_player_id
        elif match.white_player_id == current_user_id:
            opponent_id = match.black_player_id
        else:
            return False  # User không phải player trong match này
        
        if not opponent_id:
            return False  # Chưa có đối thủ (AI match hoặc PvP chưa join)
        
        # Kiểm tra nếu đối thủ có match active khác (đã tạo match mới)
        opponent_active_matches = self.get_active_matches_for_user(UUID(opponent_id))
        opponent_has_other_match = any(str(m.id) != str(match.id) for m in opponent_active_matches)
        
        if opponent_has_other_match:
            logger.info(f"Opponent {opponent_id} has other active match, considering them disconnected from match {match.id}")
            return True
        
        return False

    async def record_move(self, match: match_model.Match, move: match_schema.MoveRequest, current_user_id: Optional[str] = None) -> dict:
        """Ghi nhận nước đi và xử lý game logic.
        
        Args:
            match: Match object
            move: Move request
            current_user_id: ID của user đang submit move (optional, để check disconnect)
        """
        # RÀNG BUỘC 1: Kiểm tra match đã kết thúc chưa
        if match.finished_at is not None or match.result is not None:
            raise ValueError("Trận đấu đã kết thúc")
        
        # RÀNG BUỘC 2: Đối với PvP match, phải có cả 2 người chơi
        if not match.ai_level:
            if not match.black_player_id or not match.white_player_id:
                raise ValueError("Chưa đủ người chơi. Vui lòng đợi người chơi khác tham gia.")
        
        # RÀNG BUỘC 3: Kiểm tra user có phải là player trong match không
        if current_user_id:
            is_black = str(match.black_player_id) == str(current_user_id)
            is_white = str(match.white_player_id) == str(current_user_id) if match.white_player_id else False
            
            if not is_black and not is_white:
                raise ValueError("Bạn không phải là người chơi trong trận đấu này")
            
            # RÀNG BUỘC 4: Kiểm tra đúng lượt của người chơi (chỉ cho PvP)
            if not match.ai_level:
                # Lấy current player từ match state (an toàn hơn)
                state = await self.get_match_state(match)
                # Nếu state là None (match mới tạo), mặc định là lượt Black (người tạo bàn)
                expected_color = state.get("current_player", "B") if state else "B"
                
                # Xác định màu của user
                user_color = "B" if is_black else "W"
                
                # Kiểm tra đúng lượt
                if user_color != expected_color:
                    raise ValueError(f"Không phải lượt của bạn. Hiện tại là lượt của {expected_color}")
                
                # RÀNG BUỘC 5: Kiểm tra thời gian cho PvP matches
                if match.time_control_minutes and match.last_move_at:
                    now = datetime.now(timezone.utc)
                    elapsed_seconds = int((now - match.last_move_at).total_seconds())
                    
                    # Lấy thời gian còn lại của người chơi hiện tại
                    if user_color == "B":
                        time_remaining = match.black_time_remaining_seconds
                    else:
                        time_remaining = match.white_time_remaining_seconds
                    
                    if time_remaining is not None:
                        # Trừ thời gian đã dùng
                        new_time_remaining = time_remaining - elapsed_seconds
                        
                        # Nếu hết thời gian, tự động resign
                        if new_time_remaining <= 0:
                            # Người chơi hiện tại hết thời gian → thua
                            loser_color = user_color
                            winner_color = "W" if loser_color == "B" else "B"
                            
                            match.finished_at = now
                            match.result = f"{winner_color}+TIME"
                            self.db.commit()
                            
                            # Update Elo ratings
                            try:
                                from .statistics_service import StatisticsService
                                stats_service = StatisticsService(self.db)
                                stats_service.update_elo_ratings(match)
                            except Exception as e:
                                logger.error(f"Failed to update Elo ratings: {e}", exc_info=True)
                            
                            raise ValueError(f"Hết thời gian! Bạn thua do hết thời gian.")
                        
                        # Cập nhật thời gian còn lại
                        if user_color == "B":
                            match.black_time_remaining_seconds = new_time_remaining
                        else:
                            match.white_time_remaining_seconds = new_time_remaining
                        
                        # Cập nhật last_move_at cho nước đi mới
                        match.last_move_at = now
                        self.db.commit()
        
        # QUAN TRỌNG: Trong AI match, xác định màu của user dựa trên player_id
        # User có thể là Black hoặc White tùy thuộc vào lựa chọn khi tạo match
        if match.ai_level and current_user_id:
            # Xác định màu của user trong AI match
            user_is_black = str(match.black_player_id) == str(current_user_id) if match.black_player_id else False
            user_is_white = str(match.white_player_id) == str(current_user_id) if match.white_player_id else False
            expected_user_color = "B" if user_is_black else ("W" if user_is_white else None)
            
            if expected_user_color and move.color != expected_user_color:
                logger.warning(f"⚠️ User move color mismatch in AI match: got {move.color}, expected {expected_user_color}, forcing to {expected_user_color}")
                move.color = expected_user_color
        
        logger.debug(f"Move: {move.color} ({move.x}, {move.y}) for match {match.id}")
        
        # Check nếu đối thủ đã disconnect (chỉ cho PvP matches)
        if current_user_id and not match.ai_level:
            if self.check_opponent_disconnected(match, current_user_id):
                # Đối thủ đã disconnect → auto-resign match (đối thủ thua)
                opponent_id = match.white_player_id if match.black_player_id == current_user_id else match.black_player_id
                if opponent_id:
                    opponent = self.db.get(user_model.User, opponent_id)
                    if opponent:
                        logger.info(f"Opponent {opponent_id} disconnected, auto-resigning match {match.id}")
                        self.resign_match(match, opponent)
                        raise ValueError("Đối thủ đã rời khỏi trận đấu. Bạn thắng!")
        
        if not go:
            # Fallback nếu không có gogame_py - dùng wrapper
            # Chỉ log một lần khi bắt đầu match để tránh spam
            if not hasattr(self, '_fallback_warned_matches'):
                self._fallback_warned_matches = set()
            if match.id not in self._fallback_warned_matches:
                logger.debug(f"Fallback mode: gogame_py not available - using basic capture logic for match {match.id}")
                self._fallback_warned_matches.add(match.id)
            
            # Validate move bounds
            if move.x < 0 or move.x >= match.board_size or move.y < 0 or move.y >= match.board_size:
                raise ValueError(f"Move out of bounds: ({move.x}, {move.y}), board size: {match.board_size}")
            
            collection = self.mongo_db.get_collection("games")
            game_doc = await collection.find_one({"match_id": match.id}) or {}
            moves = game_doc.get("moves", [])
            
            # QUAN TRỌNG: Validate màu move với current_player
            current_player = game_doc.get("current_player", "B")
            if current_player != move.color:
                logger.warning(f"⚠️ Move color mismatch: current_player={current_player}, move.color={move.color}, forcing to {current_player}")
                move.color = current_player  # Force màu đúng với current_player
            
            # Lấy ko_position từ game state
            ko_position_doc = game_doc.get("ko_position")
            ko_position = None
            if ko_position_doc and isinstance(ko_position_doc, list) and len(ko_position_doc) == 2:
                ko_position = tuple(ko_position_doc)
            
            # Rebuild board_position từ moves hiện tại (trước khi thêm move mới)
            board_position_before = game_doc.get("board_position", {})
            if not board_position_before:
                # Nếu không có board_position, rebuild từ moves
                # QUAN TRỌNG: Phải rebuild đúng cách - xử lý tất cả captured stones từ tất cả moves
                board_position_before = {}
                all_captured_positions = set()  # Tập hợp tất cả vị trí đã bị bắt
                
                # Thu thập tất cả captured positions từ tất cả moves
                for move_doc in moves:
                    if move_doc.get("captured"):
                        for cx, cy in move_doc["captured"]:
                            all_captured_positions.add(f"{cx},{cy}")
                
                # Rebuild board_position: chỉ thêm moves không bị bắt
                for move_doc in moves:
                    if move_doc.get("position"):
                        x, y = move_doc["position"]
                        move_key = f"{x},{y}"
                        # Chỉ thêm nếu vị trí này không bị bắt ở move sau
                        if move_key not in all_captured_positions:
                            move_index = moves.index(move_doc) if move_doc in moves else len(moves)
                            color = "B" if move_index % 2 == 0 else "W"
                            board_position_before[move_key] = color
            
            # Add user move
            move_key = f"{move.x},{move.y}"
            
            # Validate: Vị trí phải trống (chưa có quân)
            if move_key in board_position_before:
                raise ValueError(f"Invalid move: ({move.x}, {move.y}) - position already occupied")
            
            # Validate Ko rule TRƯỚC KHI tính capture
            # QUAN TRỌNG: Luật KO (cấm cướp cờ):
            # - Nếu bạn vừa ăn 1 quân ở một điểm nào đó, đối thủ không được phép ngay lập tức 
            #   ăn lại đúng quân vừa bắt của bạn tại đúng vị trí đó trong nước tiếp theo.
            # - Họ phải đánh ở chỗ khác trước 1 nước, sau đó mới được quay lại ăn.
            # - KO rule áp dụng BẤT KỂ có capture hay không - không được đặt tại ko_position
            if ko_position and (move.x, move.y) == ko_position:
                raise ValueError(f"Invalid move: ({move.x}, {move.y}) - violates Ko rule (cannot immediately recapture at ko position)")
            
            # Tính captured stones trong fallback mode
            captured_stones = self._calculate_capture_fallback(
                board_position_before, move.x, move.y, move.color, match.board_size
            )
            
            # Validate suicide rule: Sau khi đặt quân và capture, nhóm quân mình phải còn khí
            # QUAN TRỌNG: Một nước đi "suicide" vẫn hợp lệ nếu nó dẫn đến việc ăn quân đối thủ
            # Theo luật cờ vây: Nếu nước đi dẫn đến việc ăn quân đối thủ, nó hợp lệ ngay cả khi
            # ban đầu có vẻ như là suicide (vì sau khi capture, các vị trí vừa được giải phóng
            # sẽ trở thành liberties mới cho nhóm quân mình)
            
            # Xây dựng board sau khi capture để kiểm tra
            board_after_capture = {**board_position_before, move_key: move.color}
            # Xóa captured stones - các vị trí này sẽ trở thành liberties mới
            for cx, cy in captured_stones:
                captured_key = f"{cx},{cy}"
                if captured_key in board_after_capture:
                    del board_after_capture[captured_key]
            
            # Thu thập nhóm quân mình sau khi capture (bao gồm cả các vị trí vừa được giải phóng)
            own_group_liberties = set()
            frontier = [(move.x, move.y)]
            visited = {move_key}
            
            while frontier:
                cx, cy = frontier.pop(0)
                
                # Kiểm tra 4 neighbors
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    tx, ty = cx + dx, cy + dy
                    
                    # Kiểm tra bounds
                    if tx < 0 or tx >= match.board_size or ty < 0 or ty >= match.board_size:
                        continue
                    
                    neighbor_key = f"{tx},{ty}"
                    # Kiểm tra xem vị trí này có trong board không
                    # Nếu không có trong dict → đây là vị trí trống (liberty)
                    # Nếu có trong dict nhưng giá trị là None → cũng là liberty
                    neighbor_stone = board_after_capture.get(neighbor_key)
                    
                    # Nếu là quân cùng màu (cùng nhóm) → thêm vào frontier
                    if neighbor_stone == move.color and neighbor_key not in visited:
                        visited.add(neighbor_key)
                        frontier.append((tx, ty))
                    # Nếu là vị trí trống (None hoặc không có trong dict) → đây là khí (liberty)
                    # Điều này bao gồm cả các vị trí vừa được giải phóng sau khi capture
                    elif neighbor_stone is None or neighbor_key not in board_after_capture:
                        own_group_liberties.add((tx, ty))
            
            # Nếu nhóm quân mình không còn khí → suicide (illegal)
            # NHƯNG: Nếu có captured stones, nước đi vẫn hợp lệ vì đã ăn được quân đối thủ
            # (Logic này đã được xử lý bằng cách kiểm tra liberties SAU KHI capture)
            if not own_group_liberties:
                # Nếu không có liberties VÀ không có captured stones → suicide (illegal)
                if not captured_stones:
                    raise ValueError(f"Invalid move: ({move.x}, {move.y}) - suicide move (no liberties and no capture)")
                # Nếu có captured stones nhưng vẫn không có liberties → có thể là edge case
                # Trong trường hợp này, vẫn cho phép vì đã capture được quân đối thủ
                # (Theo luật cờ vây, nếu nước đi dẫn đến capture, nó hợp lệ)
                logger.debug(f"Move ({move.x}, {move.y}) has no liberties after capture, but captured {len(captured_stones)} stones - allowing move (capture makes it legal)")
            
            # Tính ko_position mới sau move này
            new_ko_position = self._calculate_ko_position_fallback(
                board_position_before, move.x, move.y, move.color, captured_stones, match.board_size
            )
            
            # Xây dựng board_position sau khi capture
            # QUAN TRỌNG: Phải đảm bảo captured stones bị xóa khỏi board_position
            # QUAN TRỌNG: Xác định user color dựa trên player_id (user có thể là Black hoặc White)
            if match.ai_level:
                # Trong AI match, xác định user color dựa trên player_id
                user_color = "B" if match.black_player_id else "W"
            else:
                user_color = move.color
            board_position_after = {**board_position_before, move_key: user_color}
            for cx, cy in captured_stones:
                captured_key = f"{cx},{cy}"
                if captured_key in board_position_after:
                    del board_position_after[captured_key]
                    logger.debug(f"Removed captured stone from board_position: {captured_key}")
                else:
                    # Log cảnh báo nếu quân bị bắt không có trong board_position (có thể đã bị bắt trước đó)
                    logger.warning(f"Captured stone {captured_key} not found in board_position (may have been captured earlier)")
            
            # Đảm bảo tất cả captured stones đã bị xóa
            for cx, cy in captured_stones:
                captured_key = f"{cx},{cy}"
                if captured_key in board_position_after:
                    logger.error(f"❌ CRITICAL: Captured stone {captured_key} still in board_position_after! Force removing...")
                    del board_position_after[captured_key]
            
            # Tính prisoners (quân bị bắt có màu đối lập với người đánh)
            prisoners_black = game_doc.get("prisoners_black", 0)
            prisoners_white = game_doc.get("prisoners_white", 0)
            if move.color == "B":
                # Black đánh → bắt White → tăng prisoners_white
                prisoners_white += len(captured_stones)
            else:
                # White đánh → bắt Black → tăng prisoners_black
                prisoners_black += len(captured_stones)
            
            # Xây dựng board_diff
            removed_keys = [f"{cx},{cy}" for cx, cy in captured_stones]
            # Đảm bảo move_key không có trong removed_keys (không nên xảy ra, nhưng kiểm tra để an toàn)
            removed_keys = [key for key in removed_keys if key != move_key]
            board_diff = {
                "added": {move_key: move.color},
                "removed": removed_keys
            }
            logger.debug(f"Board diff: added={board_diff['added']}, removed={len(removed_keys)} stones")
            
            moves.append({
                "number": move.move_number,
                "color": move.color,
                "position": [move.x, move.y],
                "captured": captured_stones
            })
            
            await collection.update_one(
                {"match_id": match.id},
                {
                    "$setOnInsert": {"match_id": match.id, "board_size": match.board_size},
                    "$set": {
                        "moves": moves,  # Update toàn bộ moves array
                        "current_player": "W" if move.color == "B" else "B",
                        "board_position": board_position_after,  # Cập nhật board_position sau capture
                        "prisoners_black": prisoners_black,
                        "prisoners_white": prisoners_white,
                        "ko_position": list(new_ko_position) if new_ko_position else None,  # Cập nhật ko_position
                    },
                },
                upsert=True,
            )
            
            # Try AI move với wrapper
            ai_move_result = None
            game_over_after_ai = False
            if match.ai_level:
                # Xác định màu AI dựa trên player_id - AI là bên không có player_id
                ai_color = "W" if match.black_player_id else "B"
                current_player = "W" if move.color == "B" else "B"
                if current_player == ai_color:  # AI turn
                    logger.info(f"🤖 [FALLBACK] AI turn after user move (match {match.id}, level {match.ai_level}, ai_color={ai_color})")
                    logger.info(f"🤖 [FALLBACK] Wrapper available: {_GOGAME_PY_WRAPPER}")
                    ai_move_result = await self._make_ai_move_wrapper(match)
                    if ai_move_result:
                        logger.info(f"✅ [FALLBACK] AI move successful: {ai_move_result}")
                    else:
                        logger.warning(f"⚠️ [FALLBACK] AI move returned None - AI may not be available")
                    
                    # Nếu AI không thể đánh, kiểm tra consecutive passes
                    if not ai_move_result:
                        # Reload moves từ MongoDB để có moves mới nhất (đã bao gồm user move)
                        updated_game_doc = await collection.find_one({"match_id": match.id})
                        updated_moves = updated_game_doc.get("moves", []) if updated_game_doc else game_doc.get("moves", [])
                        
                        # Kiểm tra 2 move cuối có phải pass từ 2 người chơi khác nhau không
                        # QUAN TRỌNG: Phải là 2 pass từ 2 màu khác nhau (Black và White)
                        both_passed = False
                        if len(updated_moves) >= 2:
                            last_move = updated_moves[-1] if updated_moves else None
                            second_last_move = updated_moves[-2] if len(updated_moves) >= 2 else None
                            
                            if (last_move and last_move.get("position") is None and
                                second_last_move and second_last_move.get("position") is None and
                                last_move.get("color") != second_last_move.get("color")):
                                both_passed = True
                        
                        if both_passed:
                            # Cả 2 bên đều pass -> game over
                            logger.info(f"Both players passed, ending game for match {match.id}")
                            game_over_after_ai = True
                            if not match.finished_at:
                                match.finished_at = datetime.now(timezone.utc)
                                # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
                                board_position = updated_game_doc.get("board_position") if updated_game_doc else game_doc.get("board_position", {})
                                if board_position:
                                    match.result = self._calculate_game_result_fallback(board_position, match)
                                else:
                                    # Không có board_position → không thể tính điểm chính xác
                                    logger.warning(f"Cannot calculate score without board_position for match {match.id}")
                                    match.result = "DRAW"
                                self.db.commit()
            
            result = {
                "status": "accepted",
                "move": {"x": move.x, "y": move.y, "color": move.color},
                "captured": captured_stones,
                "board_diff": board_diff,
                "prisoners_black": prisoners_black,
                "prisoners_white": prisoners_white,
                "current_player": "W" if move.color == "B" else "B",
                "game_over": game_over_after_ai,
            }
            if ai_move_result:
                result["ai_move"] = ai_move_result
            return result

        # Load board từ game state
        board = await self._get_or_create_board(match)
        
        # QUAN TRỌNG: Trong AI match, validate màu move với current_player từ board
        # User có thể là Black hoặc White tùy thuộc vào lựa chọn khi tạo match
        if match.ai_level:
            current_player = board.current_player()
            # Xác định màu của user dựa trên player_id
            user_color_str = "B" if match.black_player_id else "W"
            user_color_enum = go.Color.Black if match.black_player_id else go.Color.White
            ai_color_enum = go.Color.White if match.black_player_id else go.Color.Black
            
            # Kiểm tra xem có đúng lượt của user không
            if current_player == user_color_enum:
                # User turn - force đúng màu user
                if move.color != user_color_str:
                    logger.warning(f"⚠️ Move color mismatch in AI match (gogame_py): current_player={user_color_str}, move.color={move.color}, forcing to '{user_color_str}'")
                    move.color = user_color_str  # Force màu đúng cho user
            else:
                # AI turn - không nên xảy ra vì user không thể đánh khi đến lượt AI
                logger.warning(f"⚠️ User trying to move when it's AI turn (current_player={ai_color_enum}), rejecting")
                raise ValueError("Không phải lượt của bạn")
        
        # Lưu board state trước khi apply move để tính captured stones
        board_state_before = {}
        for x in range(match.board_size):
            for y in range(match.board_size):
                stone = board.at(x, y)
                if stone != go.Stone.Empty:
                    board_state_before[f"{x},{y}"] = stone
        
        # Convert color string to enum
        color = go.Color.Black if move.color == "B" else go.Color.White
        
        # Tạo Move object
        go_move = go.Move(move.x, move.y, color)
        
        # Validate move bounds
        if move.x < 0 or move.x >= match.board_size or move.y < 0 or move.y >= match.board_size:
            raise ValueError(f"Move out of bounds: ({move.x}, {move.y}), board size: {match.board_size}")
        
        # Validate move legality
        if not board.is_legal_move(go_move):
            raise ValueError(f"Invalid move: ({move.x}, {move.y}) - illegal move (suicide or Ko)")
        
        # Apply move
        board.make_move(go_move)
        
        # Tính captured stones bằng cách so sánh board trước và sau
        captured_stones = []
        board_state_after = {}
        for x in range(match.board_size):
            for y in range(match.board_size):
                stone = board.at(x, y)
                key = f"{x},{y}"
                if stone != go.Stone.Empty:
                    board_state_after[key] = stone
                # Nếu có quân trước nhưng không có sau → bị bắt
                if key in board_state_before and key not in board_state_after:
                    captured_stones.append([x, y])
        
        # Lưu vào MongoDB
        collection = self.mongo_db.get_collection("games")
        move_doc = {
            "number": move.move_number,
            "color": move.color,
            "position": [move.x, move.y] if not go_move.is_pass else None,
            "captured": captured_stones,  # Lưu captured stones trong move
        }
        
        # Xây dựng board_position mới (chỉ quân còn lại trên bàn cờ)
        board_position = {}
        for x in range(match.board_size):
            for y in range(match.board_size):
                stone = board.at(x, y)
                if stone == go.Stone.Black:
                    board_position[f"{x},{y}"] = "B"
                elif stone == go.Stone.White:
                    board_position[f"{x},{y}"] = "W"
        
        # Đảm bảo board_position được cập nhật trong MongoDB
        await collection.update_one(
            {"match_id": match.id},
            {
                "$setOnInsert": {"match_id": match.id, "board_size": match.board_size},
                "$push": {"moves": move_doc},
                "$set": {
                    "current_player": "W" if board.current_player() == go.Color.White else "B",
                    "prisoners_black": board.get_prisoners(go.Color.Black),
                    "prisoners_white": board.get_prisoners(go.Color.White),
                    "board_position": board_position,  # Cập nhật board_position sau mỗi move
                },
            },
            upsert=True,
        )
        
        logger.debug(f"Board updated: {len(board_position)} stones, prisoners: B={board.get_prisoners(go.Color.Black)}, W={board.get_prisoners(go.Color.White)}")
        
        # Kiểm tra game over
        is_game_over = board.is_game_over()
        
        # Update match nếu game over
        if is_game_over and not match.finished_at:
            match.finished_at = datetime.now(timezone.utc)
            
            # Tính điểm và set result
            if not match.result:
                result_str = self._calculate_game_result(board, match)
                match.result = result_str
                logger.info(f"Game over for match {match.id}, result: {result_str}")
            
            self.db.commit()
            
            # Update Elo ratings nếu match kết thúc và là PvP
            if match.result and not match.ai_level:
                try:
                    from .statistics_service import StatisticsService
                    stats_service = StatisticsService(self.db)
                    stats_service.update_elo_ratings(match)
                except Exception as e:
                    logger.error(f"Failed to update Elo ratings: {e}", exc_info=True)
        
        ai_move_result = None
        
        # Nếu là AI match và chưa kết thúc, AI đi tiếp
        # Xác định màu AI dựa trên player_id - AI là bên không có player_id
        ai_color = "W" if match.black_player_id else "B"
        user_color = "B" if match.black_player_id else "W"
        print(f"🤖 AI match check: ai_color={ai_color}, user_color={user_color}, game_over={is_game_over}")
        
        if match.ai_level and not is_game_over:
            # Lấy current_player - xử lý cả trường hợp có và không có gogame_py
            if go and hasattr(board, 'current_player'):
                # Có gogame_py - dùng board.current_player()
                try:
                    current_player = board.current_player()
                    current_player_str = "W" if current_player == go.Color.White else "B"
                    is_ai_turn = current_player_str == ai_color
                    print(f"🤖 After move (gogame_py): current_player={current_player_str}, ai_color={ai_color}, is_ai_turn={is_ai_turn}")
                except Exception as e:
                    logger.warning(f"Error getting current_player from board: {e}, falling back to MongoDB state")
                    # Fallback: lấy từ MongoDB
                    collection = self.mongo_db.get_collection("games")
                    game_doc = await collection.find_one({"match_id": match.id}) or {}
                    current_player_str = game_doc.get("current_player", "B")
                    is_ai_turn = current_player_str == ai_color
                    print(f"🤖 After move (fallback from MongoDB): current_player={current_player_str}, ai_color={ai_color}, is_ai_turn={is_ai_turn}")
            else:
                # Không có gogame_py - lấy từ MongoDB
                collection = self.mongo_db.get_collection("games")
                game_doc = await collection.find_one({"match_id": match.id}) or {}
                current_player_str = game_doc.get("current_player", "B")
                is_ai_turn = current_player_str == ai_color
                print(f"🤖 After move (no gogame_py): current_player={current_player_str}, ai_color={ai_color}, is_ai_turn={is_ai_turn}")
            
            # Gọi AI khi đến lượt AI
            if is_ai_turn:
                logger.debug(f"AI turn: level {match.ai_level}")
                ai_move_result = await self._make_ai_move(match, board if go else None)
                if ai_move_result:
                    logger.debug(f"AI move: {ai_move_result.get('move', {}).get('x')}, {ai_move_result.get('move', {}).get('y')}")
                else:
                    # AI không thể đánh - tự động pass cho AI
                    logger.info(f"AI cannot move, auto-passing for match {match.id}")
                    # Tạo pass move cho AI
                    collection = self.mongo_db.get_collection("games")
                    game_doc = await collection.find_one({"match_id": match.id})
                    move_number = len((game_doc or {}).get("moves", [])) + 1
                    
                    # Xác định màu user dựa trên player_id
                    user_color_for_pass = "B" if match.black_player_id else "W"
                    
                    pass_move_doc = {
                        "number": move_number,
                        "color": ai_color,  # Sử dụng ai_color đã xác định ở trên
                        "position": None,
                        "captured": []
                    }
                    
                    moves = (game_doc or {}).get("moves", [])
                    moves.append(pass_move_doc)
                    
                    await collection.update_one(
                        {"match_id": match.id},
                        {
                            "$setOnInsert": {"match_id": match.id, "board_size": match.board_size},
                            "$set": {
                                "moves": moves,
                                "current_player": user_color_for_pass,  # Sau AI pass, đến lượt user
                            },
                        },
                        upsert=True,
                    )
                    
                    # Kiểm tra 2 pass liên tiếp từ 2 người chơi khác nhau
                    # QUAN TRỌNG: Phải là 2 pass từ 2 màu khác nhau (Black và White)
                    both_passed = False
                    if len(moves) >= 2:
                        last_move = moves[-1] if moves else None
                        second_last_move = moves[-2] if len(moves) >= 2 else None
                        
                        if (last_move and last_move.get("position") is None and
                            second_last_move and second_last_move.get("position") is None and
                            last_move.get("color") != second_last_move.get("color")):
                            both_passed = True
                    
                    # Nếu có 2 pass liên tiếp từ 2 người chơi khác nhau, kết thúc game
                    if both_passed:
                        logger.info(f"Both players passed (AI auto-pass), ending game for match {match.id}")
                        is_game_over = True
                        if not match.finished_at:
                            match.finished_at = datetime.now(timezone.utc)
                            # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
                            match.result = self._calculate_game_result(board, match)
                            self.db.commit()
                    else:
                        # AI đã pass, báo cho frontend
                        ai_move_result = {
                            "is_pass": True,
                            "board_diff": {"added": {}, "removed": []},
                            "captured": []
                        }
            else:
                logger.debug(f"Not AI turn (current: {current_player}, expected: White)")
        
        # Tính board diff (chỉ thay đổi)
        board_diff = {
            "added": {},
            "removed": []
        }
        
        # Quân mới thêm
        # QUAN TRỌNG: Đảm bảo màu trong board_diff đúng với màu đã force ở trên
        if not go_move.is_pass:
            # Trong AI match, xác định user color dựa trên player_id
            if match.ai_level:
                user_color_diff = "B" if match.black_player_id else "W"
                board_diff["added"][f"{move.x},{move.y}"] = user_color_diff  # Force đúng màu cho user
                logger.debug(f"Board diff added: {move.x},{move.y} = {user_color_diff} (forced for user in AI match)")
            else:
                board_diff["added"][f"{move.x},{move.y}"] = move.color  # PvP match
        
        # Quân bị xóa (captured)
        for x, y in captured_stones:
            board_diff["removed"].append(f"{x},{y}")
        
        # Cập nhật last_move_at sau khi move thành công (cho PvP matches với time control)
        # Lưu ý: last_move_at đã được cập nhật trong phần kiểm tra thời gian (RÀNG BUỘC 5)
        # Nhưng cần đảm bảo cập nhật cho cả gogame_py mode nếu chưa được cập nhật
        if not match.ai_level and match.time_control_minutes:
            if match.last_move_at is None or (datetime.now(timezone.utc) - match.last_move_at).total_seconds() > 60:
                # Nếu last_move_at chưa được cập nhật hoặc quá cũ, cập nhật lại
                match.last_move_at = datetime.now(timezone.utc)
                self.db.commit()
        
        result = {
            "status": "accepted",
            "move": {"x": move.x, "y": move.y, "color": move.color},
            "captured": captured_stones,  # Danh sách quân bị bắt: [[x1,y1], [x2,y2], ...]
            "board_diff": board_diff,  # Chỉ thay đổi: {added: {"x,y": "B"}, removed: ["x,y", ...]}
            "prisoners_black": board.get_prisoners(go.Color.Black),
            "prisoners_white": board.get_prisoners(go.Color.White),
            "current_player": "W" if board.current_player() == go.Color.White else "B",
            "game_over": is_game_over,
        }
        
        logger.info(f"Move accepted: captured={len(captured_stones)} stones, prisoners: B={result['prisoners_black']}, W={result['prisoners_white']}, game_over={is_game_over}")
        if captured_stones:
            logger.debug(f"Captured stones: {captured_stones}")
        
        if ai_move_result:
            result["ai_move"] = ai_move_result
            # Nếu AI pass và game over, đảm bảo game_over được set
            if ai_move_result.get("is_pass") and is_game_over:
                result["game_over"] = True
        
        return result

    async def _get_or_create_board(self, match: match_model.Match) -> "go.Board":
        """Lấy hoặc tạo Board từ game state trong MongoDB."""
        if not go:
            raise RuntimeError("gogame_py module not available")
        
        collection = self.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": match.id})
        
        board = go.Board(match.board_size)
        
        if game_doc and "moves" in game_doc:
            # Replay moves
            for move_doc in game_doc["moves"]:
                color = go.Color.Black if move_doc["color"] == "B" else go.Color.White
                if move_doc.get("position"):
                    x, y = move_doc["position"]
                    move = go.Move(x, y, color)
                else:
                    move = go.Move.Pass(color)
                
                if board.is_legal_move(move):
                    board.make_move(move)
        
        return board

    async def _make_ai_move(self, match: match_model.Match, board: "go.Board") -> dict | None:
        """AI chọn và thực hiện nước đi với timeout và error handling."""
        if not match.ai_level:
            logger.warning(f"AI level not set for match {match.id}")
            return None
        
        logger.debug(f"AI move: direct={_GOGAME_PY_DIRECT}, wrapper={_GOGAME_PY_WRAPPER}, board={board is not None}")
        
        # Try direct import first (nếu có gogame_py và board)
        if _GOGAME_PY_DIRECT and self.ai_player and go and board:
            try:
                logger.debug(f"Trying direct AI move")
                result = await self._make_ai_move_direct(match, board)
                if result:
                    return result
                logger.warning(f"Direct AI move returned None, falling back to wrapper")
            except Exception as e:
                logger.warning(f"Direct AI move failed, falling back to wrapper: {e}", exc_info=True)
        
        # Fallback to wrapper
        if _GOGAME_PY_WRAPPER:
            try:
                logger.debug(f"Using wrapper AI move")
                result = await self._make_ai_move_wrapper(match)
                if result:
                    return result
                logger.warning(f"Wrapper AI move returned None")
            except Exception as e:
                logger.error(f"Wrapper AI move failed: {e}", exc_info=True)
                return None
        
        logger.warning(f"AI not available for match {match.id} (direct={_GOGAME_PY_DIRECT}, wrapper={_GOGAME_PY_WRAPPER})")
        return None
    
    async def _make_ai_move_direct(self, match: match_model.Match, board: "go.Board") -> dict | None:
        """AI move selection với direct import."""
        timeout = self.settings.ai_move_timeout_seconds
        retry_count = self.settings.ai_move_retry_count
        
        for attempt in range(retry_count + 1):
            try:
                # Lưu board state trước khi apply AI move
                board_state_before = {}
                for x in range(match.board_size):
                    for y in range(match.board_size):
                        stone = board.at(x, y)
                        if stone != go.Stone.Empty:
                            board_state_before[f"{x},{y}"] = stone
                
                loop = asyncio.get_event_loop()
                ai_move = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.ai_player.select_move(board, match.ai_level),
                    ),
                    timeout=timeout,
                )
                
                if not ai_move or not ai_move.is_valid:
                    logger.warning(f"AI returned invalid move on attempt {attempt + 1}")
                    if attempt < retry_count:
                        continue
                    return None
                
                # Apply AI move
                board.make_move(ai_move)
                
                # Tính captured stones
                captured_stones = []
                board_state_after = {}
                for x in range(match.board_size):
                    for y in range(match.board_size):
                        stone = board.at(x, y)
                        key = f"{x},{y}"
                        if stone != go.Stone.Empty:
                            board_state_after[key] = stone
                        if key in board_state_before and key not in board_state_after:
                            captured_stones.append([x, y])
                
                # Xây dựng board_position mới
                board_position = {}
                for x in range(match.board_size):
                    for y in range(match.board_size):
                        stone = board.at(x, y)
                        if stone == go.Stone.Black:
                            board_position[f"{x},{y}"] = "B"
                        elif stone == go.Stone.White:
                            board_position[f"{x},{y}"] = "W"
                
                # Lưu AI move vào MongoDB
                collection = self.mongo_db.get_collection("games")
                game_doc = await collection.find_one({"match_id": match.id})
                move_number = len((game_doc or {}).get("moves", [])) + 1
                
                move_doc = {
                    "number": move_number,
                    "color": "W" if ai_move.color == go.Color.White else "B",
                    "position": [ai_move.x, ai_move.y] if not ai_move.is_pass else None,
                    "captured": captured_stones,
                }
                
                # Tính board diff
                board_diff = {
                    "added": {},
                    "removed": []
                }
                ai_color = "W" if ai_move.color == go.Color.White else "B"
                if not ai_move.is_pass:
                    board_diff["added"][f"{ai_move.x},{ai_move.y}"] = ai_color
                    logger.debug(f"AI board diff added: {ai_move.x},{ai_move.y} = {ai_color}")
                for x, y in captured_stones:
                    board_diff["removed"].append(f"{x},{y}")
                
                await collection.update_one(
                    {"match_id": match.id},
                    {
                        "$push": {"moves": move_doc},
                        "$set": {
                            "current_player": "B" if board.current_player() == go.Color.Black else "W",
                            "prisoners_black": board.get_prisoners(go.Color.Black),
                            "prisoners_white": board.get_prisoners(go.Color.White),
                            "board_position": board_position,
                        },
                    },
                )
                
                logger.debug(f"AI move successful (direct), level {match.ai_level}")
                return {
                    "x": ai_move.x if not ai_move.is_pass else None,
                    "y": ai_move.y if not ai_move.is_pass else None,
                    "is_pass": ai_move.is_pass,
                    "captured": captured_stones,
                    "board_diff": board_diff,
                }
                
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"AI move failed (direct) on attempt {attempt + 1}: {e}")
                if attempt < retry_count:
                    continue
                return None
        
        return None
    
    async def _make_ai_move_wrapper(self, match: match_model.Match) -> dict | None:
        """AI move selection với subprocess wrapper."""
        logger.info(f"🤖 [WRAPPER] Starting AI move wrapper for match {match.id}, level {match.ai_level}")
        logger.info(f"🤖 [WRAPPER] Wrapper available: {_GOGAME_PY_WRAPPER}")
        
        if not _GOGAME_PY_WRAPPER:
            logger.error(f"❌ [WRAPPER] AI wrapper not available - cannot make AI move")
            logger.error(f"❌ [WRAPPER] This usually means MSYS2 Python or gogame_py module is not available")
            return None
        
        # Get board state from MongoDB
        collection = self.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": match.id})
        
        if not game_doc:
            logger.error(f"❌ [WRAPPER] Game state not found for match {match.id}")
            return None
        
        board_state = {
            "board_size": match.board_size,
            "moves": game_doc.get("moves", []),
            "current_player": game_doc.get("current_player", "B"),
        }
        
        logger.info(f"🤖 [WRAPPER] Calling AI wrapper with board_state: size={board_state['board_size']}, moves={len(board_state['moves'])}, current_player={board_state['current_player']}")
        
        # Call wrapper
        try:
            loop = asyncio.get_event_loop()
            move_data = await loop.run_in_executor(
                None,
                _wrapper_select_move,
                board_state,
                match.ai_level,
            )
            
            if not move_data:
                logger.warning(f"⚠️ [WRAPPER] AI wrapper returned no move for match {match.id}")
                logger.warning(f"⚠️ [WRAPPER] Possible reasons: MSYS2 Python not found, gogame_py module not available, or AI cannot make a move")
        except Exception as e:
            logger.error(f"❌ [WRAPPER] Error calling AI wrapper: {e}", exc_info=True)
            return None
        
        if not move_data:
            logger.warning(f"⚠️ [WRAPPER] AI wrapper returned no move for match {match.id} - AI may be unable to move")
            # Nếu AI không thể đánh, có thể game đã kết thúc hoặc AI cần pass
            # Kiểm tra xem có phải game over không
            moves = game_doc.get("moves", [])
            if len(moves) >= 2:
                # Kiểm tra 2 move cuối có phải pass không
                last_two_passes = all(
                    move.get("position") is None 
                    for move in moves[-2:] if move
                )
                if last_two_passes:
                    # Cả 2 bên đều pass -> game over
                    logger.info(f"Both players passed, game should end for match {match.id}")
                    # Trả về None để báo hiệu không có move, caller sẽ xử lý game over
            return None
        
        # Nếu AI pass, chỉ cần cập nhật current_player
        if move_data.get("is_pass"):
            # Xác định màu AI
            if match.black_player_id:
                pass_ai_color = "W"  # User là black, AI là white
            else:
                pass_ai_color = "B"  # User là white, AI là black
            
            # Sau khi AI pass, đến lượt người chơi (màu đối lập với AI)
            next_player_pass = "W" if pass_ai_color == "B" else "B"
            
            move_number = len(game_doc.get("moves", [])) + 1
            move_doc = {
                "number": move_number,
                "color": pass_ai_color,
                "position": None,
            }
            
            await collection.update_one(
                {"match_id": match.id},
                {
                    "$push": {"moves": move_doc},
                    "$set": {
                        "current_player": next_player_pass,
                    },
                },
            )
            
            logger.debug(f"AI pass successful (wrapper), level {match.ai_level}")
            return {
                "x": None,
                "y": None,
                "is_pass": True,
                "color": move_data["color"],
                "board_diff": {"added": {}, "removed": []},
                "prisoners_black": game_doc.get("prisoners_black", 0),
                "prisoners_white": game_doc.get("prisoners_white", 0),
            }
        
        # AI không pass - cần tính captured stones và board_diff
        ai_x = move_data.get("x")
        ai_y = move_data.get("y")
        # Xác định màu AI dựa trên player_id - AI là bên không có player_id
        if match.black_player_id:
            ai_color = "W"  # User là black, AI là white
        else:
            ai_color = "B"  # User là white, AI là black
        print(f"🤖 [WRAPPER] AI color determined: {ai_color} (black_player_id={match.black_player_id})")
        
        if ai_x is None or ai_y is None:
            logger.error(f"AI move missing coordinates: {move_data}")
            return None
        
        # Lấy board_position hiện tại
        board_position_before = game_doc.get("board_position", {})
        
        # Tính captured stones sau khi AI đặt quân
        captured_stones = self._calculate_capture_fallback(
            board_position_before, ai_x, ai_y, ai_color, match.board_size
        )
        
        # Tạo board_diff
        board_diff = {
            "added": {f"{ai_x},{ai_y}": ai_color},
            "removed": [f"{cx},{cy}" for cx, cy in captured_stones]
        }
        
        # Cập nhật prisoners (quân bị bắt có màu đối lập với người đánh)
        prisoners_black = game_doc.get("prisoners_black", 0)
        prisoners_white = game_doc.get("prisoners_white", 0)
        
        if ai_color == "W":
            # AI (White) đánh → bắt Black → tăng prisoners_black
            prisoners_black += len(captured_stones)
        else:
            # AI (Black) đánh → bắt White → tăng prisoners_white
            prisoners_white += len(captured_stones)
        
        # Cập nhật board_position sau AI move
        board_position_after = {**board_position_before}
        board_position_after[f"{ai_x},{ai_y}"] = ai_color
        for cx, cy in captured_stones:
            captured_key = f"{cx},{cy}"
            if captured_key in board_position_after:
                del board_position_after[captured_key]
        
        # Save AI move to MongoDB
        move_number = len(game_doc.get("moves", [])) + 1
        move_doc = {
            "number": move_number,
            "color": ai_color,
            "position": [ai_x, ai_y],
            "captured": captured_stones,
        }
        
        # Sau khi AI đánh, đến lượt người chơi (màu đối lập với AI)
        next_player = "W" if ai_color == "B" else "B"
        print(f"🤖 [WRAPPER] AI move done. Next player: {next_player}")
        
        await collection.update_one(
            {"match_id": match.id},
            {
                "$push": {"moves": move_doc},
                "$set": {
                    "current_player": next_player,
                    "board_position": board_position_after,
                    "prisoners_black": prisoners_black,
                    "prisoners_white": prisoners_white,
                },
            },
        )
        
        logger.debug(f"AI move successful (wrapper), level {match.ai_level}, move: ({ai_x}, {ai_y}), captured: {len(captured_stones)}")
        return {
            "x": ai_x,
            "y": ai_y,
            "is_pass": False,
            "color": ai_color,
            "board_diff": board_diff,
            "prisoners_black": prisoners_black,
            "prisoners_white": prisoners_white,
        }

    def _calculate_territory_flood_fill(self, board: "go.Board", board_size: int) -> Tuple[int, int]:
        """Tính lãnh thổ bằng flood-fill: tìm các vùng trống được bao quanh hoàn toàn bởi một màu.
        
        Args:
            board: Board object từ gogame_py
            board_size: Kích thước bàn cờ
            
        Returns:
            Tuple (territory_black, territory_white)
        """
        territory_black = 0
        territory_white = 0
        visited = set()
        
        def flood_fill_territory(start_x: int, start_y: int) -> Tuple[set, Optional[str]]:
            """Flood-fill từ một ô trống để tìm vùng territory.
            
            Returns:
                Tuple (set of coordinates, owner) - owner là "B", "W", hoặc None nếu tranh chấp
            """
            if (start_x, start_y) in visited:
                return set(), None
            
            region = set()
            stack = [(start_x, start_y)]
            
            # Flood-fill để tìm tất cả các ô trống liên thông
            while stack:
                x, y = stack.pop()
                
                if (x, y) in visited:
                    continue
                
                # Kiểm tra ô có trống không
                if board.at(x, y) != go.Stone.Empty:
                    continue
                
                visited.add((x, y))
                region.add((x, y))
                
                # Thêm các ô trống kề bên vào stack
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < board_size and 0 <= ny < board_size):
                        neighbor_stone = board.at(nx, ny)
                        if neighbor_stone == go.Stone.Empty and (nx, ny) not in visited:
                            stack.append((nx, ny))
            
            # Sau khi flood-fill xong, kiểm tra toàn bộ biên của vùng
            # Theo luật Trung Quốc: Territory = các giao điểm trống được bao quanh hoàn toàn bởi quân của một màu
            # Lưu ý: Chỉ kiểm tra neighbors trong bàn cờ, không loại trừ vùng chạm biên nếu tất cả neighbors đều là một màu
            has_black_neighbor = False
            has_white_neighbor = False
            has_internal_neighbors = False  # Có neighbors trong bàn cờ
            
            for x, y in region:
                # Kiểm tra 4 hướng kề bên của mỗi ô trong vùng
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    
                    if not (0 <= nx < board_size and 0 <= ny < board_size):
                        # Ra ngoài bàn cờ -> bỏ qua, không ảnh hưởng đến tính territory
                        continue
                    
                    has_internal_neighbors = True
                    neighbor_stone = board.at(nx, ny)
                    
                    if neighbor_stone == go.Stone.Black:
                        has_black_neighbor = True
                    elif neighbor_stone == go.Stone.White:
                        has_white_neighbor = True
            
            # Nếu không có neighbors trong bàn cờ (toàn bộ vùng ở biên và không có quân nào kề) -> không phải territory
            if not has_internal_neighbors:
                return region, None
            
            # Theo luật Trung Quốc: Nếu chỉ có một màu neighbors (trong bàn cờ) -> là territory của màu đó
            # Không loại trừ vùng chạm biên nếu tất cả neighbors trong bàn cờ đều là một màu
            if has_black_neighbor and not has_white_neighbor:
                return region, "B"
            elif has_white_neighbor and not has_black_neighbor:
                return region, "W"
            else:
                # Có cả 2 màu hoặc không có màu nào -> không phải territory (vùng tranh chấp)
                return region, None
        
        # Duyệt qua tất cả các ô trống
        for x in range(board_size):
            for y in range(board_size):
                if (x, y) not in visited and board.at(x, y) == go.Stone.Empty:
                    region, owner = flood_fill_territory(x, y)
                    if owner == "B":
                        territory_black += len(region)
                    elif owner == "W":
                        territory_white += len(region)
        
        return territory_black, territory_white

    def _calculate_territory_flood_fill_fallback(self, board_position: dict, board_size: int) -> Tuple[int, int]:
        """Tính lãnh thổ bằng flood-fill từ board_position (fallback mode).
        
        Args:
            board_position: Dict với format {"x,y": "B"} hoặc {"x,y": "W"}
            board_size: Kích thước bàn cờ
            
        Returns:
            Tuple (territory_black, territory_white)
        """
        territory_black = 0
        territory_white = 0
        visited = set()
        
        def is_empty(x: int, y: int) -> bool:
            """Kiểm tra ô (x, y) có trống không."""
            key = f"{x},{y}"
            return board_position.get(key) is None
        
        def flood_fill_territory(start_x: int, start_y: int) -> Tuple[set, Optional[str]]:
            """Flood-fill từ một ô trống để tìm vùng territory."""
            if (start_x, start_y) in visited:
                return set(), None
            
            region = set()
            stack = [(start_x, start_y)]
            
            # Flood-fill để tìm tất cả các ô trống liên thông
            while stack:
                x, y = stack.pop()
                
                if (x, y) in visited:
                    continue
                
                # Kiểm tra ô có trống không
                if not is_empty(x, y):
                    continue
                
                visited.add((x, y))
                region.add((x, y))
                
                # Thêm các ô trống kề bên vào stack
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < board_size and 0 <= ny < board_size):
                        neighbor_key = f"{nx},{ny}"
                        neighbor_color = board_position.get(neighbor_key)
                        if neighbor_color is None and (nx, ny) not in visited:
                            stack.append((nx, ny))
            
            # Sau khi flood-fill xong, kiểm tra toàn bộ biên của vùng
            # Theo luật Trung Quốc: Territory = các giao điểm trống được bao quanh hoàn toàn bởi quân của một màu
            # Lưu ý: Chỉ kiểm tra neighbors trong bàn cờ, không loại trừ vùng chạm biên nếu tất cả neighbors đều là một màu
            has_black_neighbor = False
            has_white_neighbor = False
            has_internal_neighbors = False  # Có neighbors trong bàn cờ
            
            for x, y in region:
                # Kiểm tra 4 hướng kề bên của mỗi ô trong vùng
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    
                    if not (0 <= nx < board_size and 0 <= ny < board_size):
                        # Ra ngoài bàn cờ -> bỏ qua, không ảnh hưởng đến tính territory
                        continue
                    
                    has_internal_neighbors = True
                    neighbor_key = f"{nx},{ny}"
                    neighbor_color = board_position.get(neighbor_key)
                    
                    if neighbor_color == "B":
                        has_black_neighbor = True
                    elif neighbor_color == "W":
                        has_white_neighbor = True
            
            # Nếu không có neighbors trong bàn cờ (toàn bộ vùng ở biên và không có quân nào kề) -> không phải territory
            if not has_internal_neighbors:
                return region, None
            
            # Theo luật Trung Quốc: Nếu chỉ có một màu neighbors (trong bàn cờ) -> là territory của màu đó
            # Không loại trừ vùng chạm biên nếu tất cả neighbors trong bàn cờ đều là một màu
            if has_black_neighbor and not has_white_neighbor:
                return region, "B"
            elif has_white_neighbor and not has_black_neighbor:
                return region, "W"
            else:
                # Có cả 2 màu hoặc không có màu nào -> không phải territory (vùng tranh chấp)
                return region, None
        
        # Duyệt qua tất cả các ô trống
        for x in range(board_size):
            for y in range(board_size):
                if (x, y) not in visited and is_empty(x, y):
                    region, owner = flood_fill_territory(x, y)
                    if owner == "B":
                        territory_black += len(region)
                    elif owner == "W":
                        territory_white += len(region)
        
        return territory_black, territory_white

    def _calculate_game_result_fallback(self, board_position: dict, match: match_model.Match) -> str:
        """Tính điểm từ board_position trong fallback mode (không có gogame_py).
        
        Args:
            board_position: Dict với format {"x,y": "B"} hoặc {"x,y": "W"}
            match: Match object
            
        Returns:
            Result string theo format "B+X" hoặc "W+X" hoặc "DRAW"
        """
        # Đếm số quân còn trên bàn
        stones_black = 0
        stones_white = 0
        
        # Đếm số quân từ board_position
        for x in range(match.board_size):
            for y in range(match.board_size):
                key = f"{x},{y}"
                stone_color = board_position.get(key)
                if stone_color == "B":
                    stones_black += 1
                elif stone_color == "W":
                    stones_white += 1
        
        # Tính territory bằng flood-fill: tìm các vùng trống được bao quanh hoàn toàn bởi một màu
        territory_black, territory_white = self._calculate_territory_flood_fill_fallback(board_position, match.board_size)
        
        # Komi for white (compensation for going second) - Luật Trung Quốc: 7.5
        # Lưu ý: Komi chỉ được cộng cho White, không cộng cho Black
        komi = 7.5
        
        # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
        black_score = stones_black + territory_black
        white_score = stones_white + territory_white + komi
        
        # Log để debug
        logger.info(f"Score calculation (fallback) for match {match.id}:")
        logger.info(f"  Black: {stones_black} stones + {territory_black} territory = {black_score} points (NO KOMI)")
        logger.info(f"  White: {stones_white} stones + {territory_white} territory + {komi} komi = {white_score} points")
        
        score_diff = black_score - white_score
        logger.info(f"  Score difference: {score_diff:.1f}")
        
        if abs(score_diff) < 0.1:  # Draw (very close scores)
            return "DRAW"
        elif score_diff > 0:
            # Format: "B+{difference}({total_score})" - Black wins by difference, total score is black_score
            return f"B+{score_diff:.1f}({black_score:.1f})"
        else:
            # Format: "W+{difference}({total_score})" - White wins by difference, total score is white_score
            return f"W+{abs(score_diff):.1f}({white_score:.1f})"

    def _calculate_game_result(self, board: "go.Board", match: match_model.Match) -> str:
        """Tính điểm và trả về kết quả game.
        
        Args:
            board: Board object sau khi game kết thúc
            match: Match object
            
        Returns:
            Result string theo format "B+X" hoặc "W+X" hoặc "DRAW"
        """
        if not go:
            # Fallback: chỉ dùng prisoners từ MongoDB (synchronous, sẽ được gọi từ async context)
            # Note: Method này được gọi từ async context, nhưng MongoDB operations cần await
            # Tạm thời return simple result, sẽ được tính lại khi có go module
            logger.warning("Cannot calculate game result without gogame_py module")
            return "DRAW"
        
        # Tính điểm theo luật Trung Quốc: Điểm = Số quân còn trên bàn + Lãnh thổ + Komi
        
        # Đếm số quân còn trên bàn
        stones_black = 0
        stones_white = 0
        
        # Đếm số quân từ board
        for x in range(match.board_size):
            for y in range(match.board_size):
                stone = board.at(x, y)
                if stone == go.Stone.Black:
                    stones_black += 1
                elif stone == go.Stone.White:
                    stones_white += 1
        
        # Tính territory bằng flood-fill: tìm các vùng trống được bao quanh hoàn toàn bởi một màu
        territory_black, territory_white = self._calculate_territory_flood_fill(board, match.board_size)
        
        # Komi for white (compensation for going second) - Luật Trung Quốc: 7.5
        komi = 7.5
        
        # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
        # Lưu ý: Komi chỉ được cộng cho White, không cộng cho Black
        black_score = stones_black + territory_black
        white_score = stones_white + territory_white + komi
        
        # Log để debug
        logger.info(f"Score calculation for match {match.id}:")
        logger.info(f"  Black: {stones_black} stones + {territory_black} territory = {black_score} points (NO KOMI)")
        logger.info(f"  White: {stones_white} stones + {territory_white} territory + {komi} komi = {white_score} points")
        
        score_diff = black_score - white_score
        logger.info(f"  Score difference: {score_diff:.1f}")
        
        if abs(score_diff) < 0.1:  # Draw (very close scores)
            return "DRAW"
        elif score_diff > 0:
            # Format: "B+{difference}({total_score})" - Black wins by difference, total score is black_score
            return f"B+{score_diff:.1f}({black_score:.1f})"
        else:
            # Format: "W+{difference}({total_score})" - White wins by difference, total score is white_score
            return f"W+{abs(score_diff):.1f}({white_score:.1f})"

    def cancel_match(self, match: match_model.Match, user: user_model.User) -> bool:
        """Hủy match khi chưa có người chơi thứ 2.
        
        Args:
            match: Match object
            user: User đang yêu cầu hủy
            
        Returns:
            True nếu hủy thành công, False nếu không thể hủy
            
        Raises:
            ValueError: Nếu không phải chủ sở hữu match hoặc match đã có đủ người chơi
        """
        # Chỉ cho phép hủy PvP matches (không phải AI matches)
        if match.ai_level is not None:
            raise ValueError("Không thể hủy trận đấu với AI")
        
        # Kiểm tra user có phải là chủ sở hữu match không
        is_black_owner = match.black_player_id and str(match.black_player_id) == str(user.id)
        is_white_owner = match.white_player_id and str(match.white_player_id) == str(user.id)
        
        if not is_black_owner and not is_white_owner:
            raise ValueError("Bạn không phải là chủ sở hữu của trận đấu này")
        
        # Chỉ cho phép hủy nếu chưa có người chơi thứ 2
        if match.black_player_id and match.white_player_id:
            raise ValueError("Không thể hủy trận đấu đã có đủ người chơi. Vui lòng sử dụng chức năng đầu hàng.")
        
        # Kiểm tra match đã kết thúc chưa
        if match.finished_at or match.result:
            raise ValueError("Trận đấu đã kết thúc")
        
        # Xóa match khỏi database
        self.db.delete(match)
        self.db.commit()
        
        logger.info(f"Match {match.id} cancelled by user {user.id}")
        return True

    def resign_match(self, match: match_model.Match, resigning_user: user_model.User) -> match_model.Match:
        match.result = "W+R" if resigning_user.id == match.black_player_id else "B+R"
        match.finished_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(match)
        
        # Update Elo ratings
        try:
            from .statistics_service import StatisticsService
            stats_service = StatisticsService(self.db)
            stats_service.update_elo_ratings(match)
        except Exception as e:
            logger.error(f"Failed to update Elo ratings: {e}", exc_info=True)
        
        return match

    async def pass_turn(self, match: match_model.Match, move_number: int, color: str) -> dict:
        """Xử lý pass move."""
        if not go:
            # Fallback
            collection = self.mongo_db.get_collection("games")
            game_doc = await collection.find_one({"match_id": match.id}) or {}
            moves = game_doc.get("moves", [])
            
            moves.append({
                "number": move_number,
                "color": color,
                "position": None,
                "captured": []
            })
            
            # Kiểm tra game over (2 passes liên tiếp từ 2 người chơi khác nhau)
            # QUAN TRỌNG: Phải là 2 pass từ 2 màu khác nhau (Black và White), không phải cùng 1 người pass 2 lần
            is_game_over = False
            if len(moves) >= 2:
                # Lấy 2 move cuối cùng
                last_move = moves[-1] if moves else None
                second_last_move = moves[-2] if len(moves) >= 2 else None
                
                # Kiểm tra cả 2 đều là pass và từ 2 màu khác nhau
                if (last_move and last_move.get("position") is None and  # Last move is pass
                    second_last_move and second_last_move.get("position") is None and  # Second last is pass
                    last_move.get("color") != second_last_move.get("color")):  # Different colors
                    is_game_over = True
                    logger.info(f"Game over: Both players passed consecutively (fallback mode, match {match.id})")
            
            await collection.update_one(
                {"match_id": match.id},
                {
                    "$setOnInsert": {"match_id": match.id, "board_size": match.board_size},
                    "$set": {
                        "moves": moves,
                        "current_player": "W" if color == "B" else "B",
                    },
                },
                upsert=True,
            )
            
            # Update match nếu game over
            if is_game_over and not match.finished_at:
                match.finished_at = datetime.now(timezone.utc)
                # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
                board_position = game_doc.get("board_position", {})
                if board_position:
                    match.result = self._calculate_game_result_fallback(board_position, match)
                else:
                    # Không có board_position → không thể tính điểm chính xác
                    logger.warning(f"Cannot calculate score without board_position for match {match.id}")
                    match.result = "DRAW"
                self.db.commit()
            
            new_current_player = "W" if color == "B" else "B"
            result = {
                "status": "pass-recorded", 
                "game_over": is_game_over,
                "current_player": new_current_player
            }
            
            # Nếu chưa game over và là AI match, thử gọi AI move (fallback)
            if not is_game_over and match.ai_level:
                # Xác định màu AI và user dựa trên player_id
                ai_color_pass = "W" if match.black_player_id else "B"
                user_color_pass = "B" if match.black_player_id else "W"
                
                if new_current_player == ai_color_pass:  # AI turn
                    logger.debug(f"AI turn after user pass (fallback), ai_color={ai_color_pass}")
                    try:
                        ai_move_result = await self._make_ai_move_wrapper(match)
                        if ai_move_result:
                            result["ai_move"] = ai_move_result
                            # Cập nhật current_player sau AI move
                            result["current_player"] = user_color_pass  # Sau AI move, đến lượt user
                            
                            # Nếu AI pass, kiểm tra lại consecutive passes
                            if ai_move_result.get("is_pass"):
                                # Reload moves để có moves mới nhất (bao gồm AI pass)
                                updated_game_doc = await collection.find_one({"match_id": match.id})
                                updated_moves = updated_game_doc.get("moves", []) if updated_game_doc else moves
                                
                                # Kiểm tra lại consecutive passes sau AI pass
                                # QUAN TRỌNG: Phải là 2 pass từ 2 màu khác nhau
                                both_passed_after_ai = False
                                if len(updated_moves) >= 2:
                                    last_move_ai = updated_moves[-1] if updated_moves else None
                                    second_last_move_ai = updated_moves[-2] if len(updated_moves) >= 2 else None
                                    
                                    if (last_move_ai and last_move_ai.get("position") is None and
                                        second_last_move_ai and second_last_move_ai.get("position") is None and
                                        last_move_ai.get("color") != second_last_move_ai.get("color")):
                                        both_passed_after_ai = True
                                
                                if both_passed_after_ai:
                                    # Cả 2 bên đều pass -> game over
                                    logger.info(f"Both players passed after AI pass, ending game for match {match.id}")
                                    result["game_over"] = True
                                    if not match.finished_at:
                                        match.finished_at = datetime.now(timezone.utc)
                                        # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
                                        board_position = updated_game_doc.get("board_position") if updated_game_doc else game_doc.get("board_position", {})
                                        if board_position:
                                            match.result = self._calculate_game_result_fallback(board_position, match)
                                        else:
                                            # Không có board_position → không thể tính điểm chính xác
                                            logger.warning(f"Cannot calculate score without board_position for match {match.id}")
                                            match.result = "DRAW"
                                        self.db.commit()
                        else:
                            # AI không thể đánh - có thể game đã kết thúc
                            logger.warning(f"AI cannot move after user pass for match {match.id}")
                    except Exception as e:
                        logger.warning(f"Failed to make AI move after pass: {e}")
            
            return result

        board = await self._get_or_create_board(match)
        go_color = go.Color.Black if color == "B" else go.Color.White
        pass_move = go.Move.Pass(go_color)
        
        if not board.is_legal_move(pass_move):
            raise ValueError("Invalid pass move")
        
        board.make_move(pass_move)
        
        # Lưu vào MongoDB
        collection = self.mongo_db.get_collection("games")
        await collection.update_one(
            {"match_id": match.id},
            {
                "$setOnInsert": {"match_id": match.id, "board_size": match.board_size},
                "$push": {"moves": {"number": move_number, "color": color, "position": None}},
                "$set": {
                    "current_player": "W" if board.current_player() == go.Color.White else "B",
                },
            },
            upsert=True,
        )
        
        # Reload moves từ MongoDB để có moves mới nhất (bao gồm pass vừa thêm)
        updated_game_doc = await collection.find_one({"match_id": match.id})
        updated_moves = updated_game_doc.get("moves", []) if updated_game_doc else []
        
        # Kiểm tra game over (2 passes liên tiếp từ 2 người chơi khác nhau)
        # QUAN TRỌNG: Phải là 2 pass từ 2 màu khác nhau (Black và White), không phải cùng 1 người pass 2 lần
        is_game_over = False
        if len(updated_moves) >= 2:
            # Lấy 2 move cuối cùng
            last_move = updated_moves[-1] if updated_moves else None
            second_last_move = updated_moves[-2] if len(updated_moves) >= 2 else None
            
            # Kiểm tra cả 2 đều là pass và từ 2 màu khác nhau
            if (last_move and last_move.get("position") is None and  # Last move is pass
                second_last_move and second_last_move.get("position") is None and  # Second last is pass
                last_move.get("color") != second_last_move.get("color")):  # Different colors
                is_game_over = True
                logger.info(f"Game over: Both players passed consecutively (PvP match {match.id})")
        
        # Nếu kiểm tra thủ công phát hiện game over nhưng board.is_game_over() chưa phát hiện
        # thì vẫn dùng kết quả từ kiểm tra thủ công
        if is_game_over and not board.is_game_over():
            logger.info(f"Manual check detected game over (2 consecutive passes) for match {match.id}, but board.is_game_over() returned False")
        
        # Update match nếu game over
        if is_game_over and not match.finished_at:
            match.finished_at = datetime.now(timezone.utc)
            
            # Tính điểm và set result
            if not match.result:
                result_str = self._calculate_game_result(board, match)
                match.result = result_str
                logger.info(f"Game over for match {match.id} (pass), result: {result_str}")
            
            self.db.commit()
            
            # Update Elo ratings nếu match kết thúc và là PvP
            if match.result and not match.ai_level:
                try:
                    from .statistics_service import StatisticsService
                    stats_service = StatisticsService(self.db)
                    stats_service.update_elo_ratings(match)
                except Exception as e:
                    logger.error(f"Failed to update Elo ratings: {e}", exc_info=True)
        
        # Nếu chưa game over và là AI match, tự động gọi AI move
        ai_move_result = None
        if not is_game_over and match.ai_level:
            current_player = board.current_player()
            # Xác định màu AI dựa trên player_id
            ai_color_enum = go.Color.White if match.black_player_id else go.Color.Black
            # Nếu đến lượt AI sau khi user pass
            if current_player == ai_color_enum:
                logger.debug(f"AI turn after user pass (ai_color={ai_color_enum})")
                ai_move_result = await self._make_ai_move(match, board)
                
                # Nếu AI pass, kiểm tra lại consecutive passes
                if ai_move_result and ai_move_result.get("is_pass"):
                    # Reload moves từ MongoDB để có moves mới nhất (bao gồm AI pass)
                    final_game_doc = await collection.find_one({"match_id": match.id})
                    final_moves = final_game_doc.get("moves", []) if final_game_doc else updated_moves
                    
                    # Kiểm tra lại consecutive passes sau AI pass
                    # QUAN TRỌNG: Phải là 2 pass từ 2 màu khác nhau
                    consecutive_passes_after_ai = False
                    if len(final_moves) >= 2:
                        last_move_ai = final_moves[-1] if final_moves else None
                        second_last_move_ai = final_moves[-2] if len(final_moves) >= 2 else None
                        
                        if (last_move_ai and last_move_ai.get("position") is None and
                            second_last_move_ai and second_last_move_ai.get("position") is None and
                            last_move_ai.get("color") != second_last_move_ai.get("color")):
                            consecutive_passes_after_ai = True
                    
                    if consecutive_passes_after_ai:
                        # Cả 2 bên đều pass -> game over
                        logger.info(f"Both players passed (user + AI), ending game for match {match.id}")
                        is_game_over = True
                        if not match.finished_at:
                            match.finished_at = datetime.now(timezone.utc)
                            # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
                            match.result = self._calculate_game_result(board, match)
                            self.db.commit()
        
        result = {
            "status": "pass-recorded", 
            "game_over": is_game_over,
            "current_player": "W" if board.current_player() == go.Color.White else "B"
        }
        if ai_move_result:
            result["ai_move"] = ai_move_result
        
        return result

    async def export_sgf(self, match: match_model.Match) -> str:
        """Export match sang SGF format.
        
        Args:
            match: Match object
        
        Returns:
            SGF string
        """
        from ..tasks import background
        
        # Lấy game state từ MongoDB
        collection = self.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": match.id})
        
        if not game_doc:
            raise ValueError("Match không có game state")
        
        moves = game_doc.get("moves", [])
        
        # Lấy player names nếu có
        black_player = None
        white_player = None
        if match.black_player_id:
            black_user = self.db.get(user_model.User, match.black_player_id)
            if black_user:
                black_player = black_user.username
        if match.white_player_id:
            white_user = self.db.get(user_model.User, match.white_player_id)
            if white_user:
                white_player = white_user.username
        
        # Export SGF trong background
        sgf_content = await background.export_match_sgf(
            match_id=str(match.id),
            moves=moves,
            board_size=match.board_size,
        )
        
        # Update match với SGF ID
        match.sgf_id = str(match.id)  # Hoặc có thể dùng SGF file ID
        self.db.commit()
        
        return sgf_content

    async def import_sgf(self, user: user_model.User, sgf_content: str) -> match_model.Match:
        """Import game từ SGF format.
        
        Args:
            user: User importing the game
            sgf_content: SGF format string
        
        Returns:
            Created Match object
        """
        from ..utils.sgf import parse_sgf
        
        # Parse SGF
        game_data = parse_sgf(sgf_content)
        
        # Create match
        match = match_model.Match(
            black_player_id=user.id,  # User is always Black when importing
            white_player_id=None,  # No opponent for imported games
            ai_level=None,
            board_size=game_data["board_size"],
            result=game_data.get("result"),
            started_at=game_data.get("date") or datetime.now(timezone.utc),
            finished_at=game_data.get("date") if game_data.get("result") else None,
        )
        self.db.add(match)
        self.db.commit()
        self.db.refresh(match)
        
        # Save game state to MongoDB
        collection = self.mongo_db.get_collection("games")
        await collection.insert_one({
            "match_id": match.id,
            "board_size": game_data["board_size"],
            "moves": game_data["moves"],
            "current_player": "B" if len(game_data["moves"]) % 2 == 0 else "W",
            "prisoners_black": 0,
            "prisoners_white": 0,
        })
        
        logger.info(f"Imported SGF game: {match.id}, {len(game_data['moves'])} moves")
        return match

    async def undo_move(self, match: match_model.Match, current_user_id: str) -> dict:
        """Hoàn tác nước đi cuối cùng.
        
        Args:
            match: Match object
            current_user_id: ID của user đang yêu cầu undo
        
        Returns:
            Dict với thông tin về move đã undo và board state mới
        
        Raises:
            ValueError: Nếu không thể undo (match ended, no moves, not user's move, etc.)
        """
        # Kiểm tra match chưa kết thúc
        if match.finished_at:
            raise ValueError("Không thể undo: Trận đấu đã kết thúc")
        
        # Lấy game state từ MongoDB
        collection = self.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": match.id})
        
        if not game_doc:
            raise ValueError("Không tìm thấy game state")
        
        moves = game_doc.get("moves", [])
        
        # Kiểm tra có moves không
        if not moves:
            raise ValueError("Không có nước đi nào để undo")
        
        # Lấy move cuối cùng
        last_move = moves[-1]
        last_move_color = last_move.get("color")
        
        # Kiểm tra move cuối cùng là của user
        # Trong AI match: user là Black, AI là White
        # Trong PvP match: kiểm tra user có phải là player của màu đó không
        moves_to_undo = []  # Danh sách moves cần undo
        
        if match.ai_level:
            # AI match: Logic đặc biệt
            # - Nếu move cuối cùng là của AI (White) → undo cả AI move và user move trước đó
            # - Nếu move cuối cùng là của user (Black) → chỉ undo user move
            if last_move_color == "W":
                # Move cuối cùng là của AI → undo cả AI và user move trước đó
                if len(moves) < 2:
                    raise ValueError("Không thể undo: Cần ít nhất 2 nước đi (user + AI)")
                
                # Undo AI move (cuối cùng)
                ai_move = moves.pop()
                moves_to_undo.append(ai_move)
                
                # Undo user move (trước đó)
                user_move = moves.pop()
                moves_to_undo.append(user_move)
                
                logger.info(f"Undo AI match: Removed AI move ({ai_move.get('position')}) and user move ({user_move.get('position')})")
            else:
                # Move cuối cùng là của user → chỉ undo user move
                user_move = moves.pop()
                moves_to_undo.append(user_move)
                logger.info(f"Undo AI match: Removed user move ({user_move.get('position')})")
        else:
            # PvP match: kiểm tra user có phải là player của màu đó không
            is_black_player = match.black_player_id == current_user_id
            is_white_player = match.white_player_id == current_user_id
            
            if last_move_color == "B" and not is_black_player:
                raise ValueError("Không thể undo: Nước đi cuối cùng không phải của bạn")
            if last_move_color == "W" and not is_white_player:
                raise ValueError("Không thể undo: Nước đi cuối cùng không phải của bạn")
            
            # Chỉ undo 1 move trong PvP
            user_move = moves.pop()
            moves_to_undo.append(user_move)
        
        # Rebuild board state từ moves còn lại
        if not go:
            # Fallback mode: rebuild từ moves
            # QUAN TRỌNG: Tính lại prisoners từ đầu từ moves còn lại
            # Không trừ từ prisoners hiện tại để tránh sai sót tích lũy
            board_position = {}
            prisoners_black = 0
            prisoners_white = 0
            current_player = "B"  # Default nếu không có moves
            
            # Rebuild board từ moves còn lại và tính lại prisoners từ đầu
            for move_doc in moves:
                move_color = move_doc.get("color")
                move_pos = move_doc.get("position")
                captured = move_doc.get("captured", [])
                
                # Thêm quân mới vào board
                if move_pos:
                    x, y = move_pos
                    key = f"{x},{y}"
                    board_position[key] = move_color
                
                # Xóa captured stones khỏi board_position
                for cap in captured:
                    if isinstance(cap, list) and len(cap) == 2:
                        cap_x, cap_y = cap
                        cap_key = f"{cap_x},{cap_y}"
                        if cap_key in board_position:
                            del board_position[cap_key]
                
                # Tính lại prisoners từ đầu (quân bị bắt có màu đối lập với người đánh)
                for cap in captured:
                    if isinstance(cap, list) and len(cap) == 2:
                        if move_color == "B":
                            prisoners_white += 1  # Black bắt White → tăng prisoners_white
                        else:
                            prisoners_black += 1  # White bắt Black → tăng prisoners_black
                
                # Cập nhật current_player
                current_player = "W" if move_color == "B" else "B"
            
            # Tính ko_position từ move trước đó (nếu có)
            ko_position = None
            if len(moves) >= 1:
                prev_move = moves[-1]
                prev_captured = prev_move.get("captured", [])
                prev_pos = prev_move.get("position")
                
                # Ko rule: capture đúng 1 quân và nhóm mình chỉ có 1 quân
                if len(prev_captured) == 1 and prev_pos:
                    prev_x, prev_y = prev_pos
                    prev_key = f"{prev_x},{prev_y}"
                    
                    # Kiểm tra xem nhóm quân tại prev_pos có chỉ 1 quân không
                    # (đơn giản hóa: nếu không có quân cùng màu kề bên → nhóm 1 quân)
                    prev_color = prev_move.get("color")
                    has_neighbor = False
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = prev_x + dx, prev_y + dy
                        if 0 <= nx < match.board_size and 0 <= ny < match.board_size:
                            neighbor_key = f"{nx},{ny}"
                            if board_position.get(neighbor_key) == prev_color:
                                has_neighbor = True
                                break
                    
                    if not has_neighbor:
                        # Nhóm chỉ có 1 quân → có ko_position
                        if isinstance(prev_captured[0], list) and len(prev_captured[0]) == 2:
                            ko_position = prev_captured[0]
            
            # Cập nhật MongoDB
            await collection.update_one(
                {"match_id": match.id},
                {
                    "$set": {
                        "moves": moves,
                        "board_position": board_position,
                        "current_player": current_player,
                        "prisoners_black": prisoners_black,
                        "prisoners_white": prisoners_white,
                        "ko_position": list(ko_position) if ko_position else None,
                    }
                }
            )
            
            return {
                "status": "undone",
                "undone_moves": moves_to_undo,  # Danh sách moves đã undo
                "undone_move": moves_to_undo[0] if moves_to_undo else None,  # Giữ backward compatibility
                "board_position": board_position,
                "current_player": current_player,
                "prisoners_black": prisoners_black,
                "prisoners_white": prisoners_white,
                "remaining_moves": len(moves),
            }
        
        # gogame_py mode: rebuild board từ moves
        board = go.Board(match.board_size)
        
        # Apply tất cả moves trừ move cuối cùng
        for move_doc in moves:
            move_color = move_doc.get("color")
            move_pos = move_doc.get("position")
            
            if not move_pos:
                # Pass move
                color = go.Color.Black if move_color == "B" else go.Color.White
                pass_move = go.Move.Pass(color)
                board.make_move(pass_move)
            else:
                x, y = move_pos
                color = go.Color.Black if move_color == "B" else go.Color.White
                go_move = go.Move(x, y, color)
                board.make_move(go_move)
        
        # Xây dựng board_position từ board
        board_position = {}
        for x in range(match.board_size):
            for y in range(match.board_size):
                stone = board.at(x, y)
                if stone == go.Stone.Black:
                    board_position[f"{x},{y}"] = "B"
                elif stone == go.Stone.White:
                    board_position[f"{x},{y}"] = "W"
        
        # Tính ko_position từ board (nếu có)
        # Lưu ý: gogame_py Board có thể không expose ko_index trực tiếp
        # Tạm thời set None, sẽ được tính lại khi có move tiếp theo
        ko_position = None
        # TODO: Có thể cần thêm method để lấy ko_index từ board nếu cần
        
        # Cập nhật MongoDB
        await collection.update_one(
            {"match_id": match.id},
            {
                "$set": {
                    "moves": moves,
                    "board_position": board_position,
                    "current_player": "B" if board.current_player() == go.Color.Black else "W",
                    "prisoners_black": board.get_prisoners(go.Color.Black),
                    "prisoners_white": board.get_prisoners(go.Color.White),
                    "ko_position": ko_position,
                }
            }
        )
        
        return {
            "status": "undone",
            "undone_moves": moves_to_undo,  # Danh sách moves đã undo
            "undone_move": moves_to_undo[0] if moves_to_undo else None,  # Giữ backward compatibility
            "board_position": board_position,
            "current_player": "B" if board.current_player() == go.Color.Black else "W",
            "prisoners_black": board.get_prisoners(go.Color.Black),
            "prisoners_white": board.get_prisoners(go.Color.White),
            "remaining_moves": len(moves),
        }

    async def get_replay(self, match: match_model.Match) -> dict:
        """Lấy replay data cho match.
        
        Args:
            match: Match object
        
        Returns:
            Dict với replay data
        
        Raises:
            ValueError: Nếu match không có game state
        """
        collection = self.mongo_db.get_collection("games")
        game_doc = await collection.find_one({"match_id": str(match.id)})  # Ensure string match_id
        
        if not game_doc:
            # Try with UUID format
            game_doc = await collection.find_one({"match_id": match.id})
            if not game_doc:
                raise ValueError("Match không có game state")
        
        moves = game_doc.get("moves", [])
        
        # Get player names
        black_player = None
        white_player = None
        if match.black_player_id:
            black_user = self.db.get(user_model.User, match.black_player_id)
            if black_user:
                black_player = black_user.username
        if match.white_player_id:
            white_user = self.db.get(user_model.User, match.white_player_id)
            if white_user:
                white_player = white_user.username
        
        # Get prisoners from game_doc or calculate from moves
        prisoners_black = game_doc.get("prisoners_black", 0)
        prisoners_white = game_doc.get("prisoners_white", 0)
        
        return {
            "match_id": str(match.id),
            "board_size": match.board_size,
            "black_player": black_player,
            "white_player": white_player,
            "result": match.result,
            "moves": moves,
            "total_moves": len(moves),
            "prisoners_black": prisoners_black,
            "prisoners_white": prisoners_white,
            "current_player": game_doc.get("current_player", "B"),
        }

