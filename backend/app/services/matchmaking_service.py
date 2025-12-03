"""Service cho matchmaking (ghép người chơi online)."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from ..models.sql import match as match_model
from ..models.sql import user as user_model

logger = logging.getLogger(__name__)

# Matchmaking configuration
ELO_RANGE_INITIAL = 200  # Initial ELO range for matching (±200)
ELO_RANGE_EXPANSION = 50  # Expand range by 50 every 5 seconds
MAX_ELO_DIFFERENCE = 500  # Maximum ELO difference allowed
QUEUE_TIMEOUT = 60  # Timeout after 60 seconds in queue
MATCHING_INTERVAL = 2  # Check for matches every 2 seconds


class QueueEntry:
    """Entry trong matchmaking queue."""
    
    def __init__(self, user_id: str, elo_rating: int, board_size: int, joined_at: datetime):
        self.user_id = user_id
        self.elo_rating = elo_rating
        self.board_size = board_size
        self.joined_at = joined_at
        self.elo_range = ELO_RANGE_INITIAL
    
    def expand_elo_range(self):
        """Mở rộng ELO range theo thời gian chờ."""
        elapsed = (datetime.now(timezone.utc) - self.joined_at).total_seconds()
        expansions = int(elapsed / 5)  # Expand every 5 seconds
        self.elo_range = min(
            ELO_RANGE_INITIAL + (expansions * ELO_RANGE_EXPANSION),
            MAX_ELO_DIFFERENCE
        )
    
    def is_compatible(self, other: QueueEntry) -> bool:
        """Kiểm tra xem có thể match với entry khác không."""
        if self.board_size != other.board_size:
            return False
        
        # Check ELO compatibility (bidirectional)
        elo_diff = abs(self.elo_rating - other.elo_rating)
        return (
            elo_diff <= self.elo_range and
            elo_diff <= other.elo_range
        )


class MatchmakingService:
    """Service quản lý matchmaking queue và matching algorithm.
    
    Singleton pattern: Queue và matching thread được chia sẻ giữa tất cả instances.
    """
    
    # Class-level shared state (singleton pattern)
    _shared_queue: Dict[int, List[QueueEntry]] = {}  # board_size -> [QueueEntry]
    _shared_matching_task: Optional[asyncio.Task] = None
    _shared_matching_thread: Optional[threading.Thread] = None
    _shared_running = False
    _shared_loop: Optional[asyncio.AbstractEventLoop] = None
    _shared_lock = threading.Lock()
    
    def __init__(self, db: Session):
        self.db = db
        # Sử dụng shared state thay vì instance variables
    
    def start_matching_task(self):
        """Bắt đầu background task để match players."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 [START_MATCHING] start_matching_task() called")
        print("🚀 [START_MATCHING] start_matching_task() called")
        
        try:
            logger.info("🔒 [START_MATCHING] Acquiring lock...")
            print("🔒 [START_MATCHING] Acquiring lock...")
            with MatchmakingService._shared_lock:
                logger.info("✅ [START_MATCHING] Lock acquired")
                print("✅ [START_MATCHING] Lock acquired")
                
                if not MatchmakingService._shared_running:
                    logger.info("🔄 [START_MATCHING] Setting _shared_running = True")
                    print("🔄 [START_MATCHING] Setting _shared_running = True")
                    MatchmakingService._shared_running = True
                    
                    # Start matching loop in a separate thread with its own event loop
                    logger.info("🧵 [START_MATCHING] Creating matching thread...")
                    print("🧵 [START_MATCHING] Creating matching thread...")
                    MatchmakingService._shared_matching_thread = threading.Thread(
                        target=self._run_matching_loop,
                        daemon=True,
                        name="MatchmakingThread"
                    )
                    
                    logger.info("🚀 [START_MATCHING] Starting thread...")
                    print("🚀 [START_MATCHING] Starting thread...")
                    MatchmakingService._shared_matching_thread.start()
                    
                    logger.info("✅ [START_MATCHING] Matchmaking service started successfully")
                    print("✅ [START_MATCHING] Matchmaking service started successfully")
                else:
                    logger.info("ℹ️ [START_MATCHING] Matching task already running, skipping")
                    print("ℹ️ [START_MATCHING] Matching task already running, skipping")
        except Exception as e:
            logger.error(f"❌ [START_MATCHING] Error in start_matching_task: {e}", exc_info=True)
            print(f"❌ [START_MATCHING] Error in start_matching_task: {e}")
            raise
    
    def _run_matching_loop(self):
        """Chạy matching loop trong một thread riêng với event loop riêng."""
        logger.info("🚀 Starting matching thread...")
        # Tạo event loop mới cho thread này
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        MatchmakingService._shared_loop = loop
        
        try:
            logger.info("✅ Matching thread event loop created, starting matching loop...")
            loop.run_until_complete(self._matching_loop())
        except asyncio.CancelledError:
            logger.info("Matchmaking loop cancelled")
        except Exception as e:
            logger.error(f"❌ Fatal error in matching thread: {e}", exc_info=True)
        finally:
            logger.info("🛑 Matching thread stopping, closing event loop...")
            loop.close()
            MatchmakingService._shared_loop = None
            MatchmakingService._shared_running = False
    
    def stop_matching_task(self):
        """Dừng background task."""
        with MatchmakingService._shared_lock:
            if not MatchmakingService._shared_running:
                return  # Already stopped
            
            logger.info("🛑 [MATCHMAKING] Stopping matching task...")
            MatchmakingService._shared_running = False
            
            # Cancel matching task nếu có
            if MatchmakingService._shared_loop and MatchmakingService._shared_matching_task:
                try:
                    MatchmakingService._shared_loop.call_soon_threadsafe(
                        MatchmakingService._shared_matching_task.cancel
                    )
                    logger.info("🛑 [MATCHMAKING] Task cancellation scheduled")
                except Exception as e:
                    logger.warning(f"⚠️ [MATCHMAKING] Error cancelling matching task: {e}")
            
            # Stop event loop
            if MatchmakingService._shared_loop:
                try:
                    if MatchmakingService._shared_loop.is_running():
                        MatchmakingService._shared_loop.call_soon_threadsafe(MatchmakingService._shared_loop.stop)
                        logger.info("🛑 [MATCHMAKING] Event loop stop scheduled")
                except Exception as e:
                    logger.warning(f"⚠️ [MATCHMAKING] Error stopping event loop: {e}")
            
            # Wait for thread to finish
            if MatchmakingService._shared_matching_thread and MatchmakingService._shared_matching_thread.is_alive():
                logger.info("🛑 [MATCHMAKING] Waiting for thread to stop (timeout: 3s)...")
                try:
                    MatchmakingService._shared_matching_thread.join(timeout=3.0)
                    if MatchmakingService._shared_matching_thread.is_alive():
                        logger.warning("⚠️ [MATCHMAKING] Thread did not stop within timeout (daemon thread will exit with main process)")
                    else:
                        logger.info("✅ [MATCHMAKING] Thread stopped successfully")
                except Exception as e:
                    logger.warning(f"⚠️ [MATCHMAKING] Error joining matching thread: {e}")
            
            # Clean up references
            MatchmakingService._shared_matching_task = None
            MatchmakingService._shared_matching_thread = None
            MatchmakingService._shared_loop = None
            logger.info("✅ [MATCHMAKING] Matchmaking service stopped")
    
    async def _matching_loop(self):
        """Background loop để match players."""
        try:
            while MatchmakingService._shared_running:
                try:
                    self._try_match_players()  # Synchronous function, không cần await
                    await asyncio.sleep(MATCHING_INTERVAL)
                except asyncio.CancelledError:
                    logger.info("Matching loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in matching loop: {e}", exc_info=True)
                    # Continue running even if there's an error
                    try:
                        await asyncio.sleep(MATCHING_INTERVAL)
                    except asyncio.CancelledError:
                        break
        except asyncio.CancelledError:
            logger.info("Matching loop cancelled (outer)")
        except Exception as e:
            logger.error(f"Fatal error in matching loop: {e}", exc_info=True)
    
    def _try_match_players(self):
        """Thử match players trong queue.
        
        Chỉ match những người chơi có cùng board_size.
        Queue được tổ chức theo board_size, mỗi board_size có queue riêng.
        """
        try:
            with MatchmakingService._shared_lock:
                for board_size, queue in list(MatchmakingService._shared_queue.items()):
                    if len(queue) < 2:
                        continue
                    
                    # Validate: tất cả entries trong queue phải có cùng board_size
                    for entry in queue:
                        if entry.board_size != board_size:
                            logger.error(
                                f"Queue inconsistency: entry {entry.user_id} has board_size {entry.board_size} "
                                f"but queue is for board_size {board_size}"
                            )
                            # Remove invalid entry
                            queue.remove(entry)
                            continue
                    
                    if len(queue) < 2:
                        continue
                    
                    # Expand ELO ranges for all entries
                    for entry in queue:
                        entry.expand_elo_range()
                    
                    # Try to find matches
                    matched_pairs = []
                    used_indices = set()
                    
                    for i, entry1 in enumerate(queue):
                        if i in used_indices:
                            continue
                        
                        # Đảm bảo entry1 có đúng board_size
                        if entry1.board_size != board_size:
                            logger.warning(f"Skipping entry1 {entry1.user_id} - wrong board_size")
                            continue
                        
                        for j, entry2 in enumerate(queue[i+1:], start=i+1):
                            if j in used_indices:
                                continue
                            
                            # Đảm bảo entry2 có đúng board_size
                            if entry2.board_size != board_size:
                                logger.warning(f"Skipping entry2 {entry2.user_id} - wrong board_size")
                                continue
                            
                            # Double-check board_size compatibility
                            if entry1.board_size != entry2.board_size:
                                logger.error(
                                    f"Board size mismatch: {entry1.user_id} ({entry1.board_size}) "
                                    f"vs {entry2.user_id} ({entry2.board_size})"
                                )
                                continue
                            
                            if entry1.is_compatible(entry2):
                                matched_pairs.append((i, j))
                                used_indices.add(i)
                                used_indices.add(j)
                                logger.info(
                                    f"Found compatible pair: {entry1.user_id} (ELO {entry1.elo_rating}, "
                                    f"board {entry1.board_size}) vs {entry2.user_id} (ELO {entry2.elo_rating}, "
                                    f"board {entry2.board_size})"
                                )
                                break
                    
                    # Tạo matches TRƯỚC KHI remove khỏi queue để đảm bảo match được tạo thành công
                    # Nếu match creation fail, entries vẫn còn trong queue
                    matched_entries = []
                    for i, j in sorted(matched_pairs, reverse=True):
                        # Lấy entries nhưng chưa pop khỏi queue
                        entry1 = queue[max(i, j)]
                        entry2 = queue[min(i, j)]
                        
                        # Final validation: đảm bảo board_size khớp
                        if entry1.board_size != entry2.board_size or entry1.board_size != board_size:
                            logger.error(
                                f"Board size mismatch when creating match: "
                                f"entry1.board_size={entry1.board_size}, "
                                f"entry2.board_size={entry2.board_size}, "
                                f"queue board_size={board_size}"
                            )
                            continue
                        
                        matched_entries.append((i, j, entry1, entry2, board_size))
                        logger.info(
                            f"Found match: {entry1.user_id} (ELO {entry1.elo_rating}) "
                            f"vs {entry2.user_id} (ELO {entry2.elo_rating}) "
                            f"on {board_size}x{board_size} board"
                        )
                    
                    # Create matches TRƯỚC KHI remove khỏi queue
                    successful_matches = []
                    for i, j, entry1, entry2, bs in matched_entries:
                        # Final check trước khi tạo match
                        if entry1.board_size != bs or entry2.board_size != bs:
                            logger.error(
                                f"Cannot create match - board size mismatch: "
                                f"entry1={entry1.board_size}, entry2={entry2.board_size}, expected={bs}"
                            )
                            continue
                        
                        try:
                            # Tạo match synchronously để đảm bảo nó được tạo ngay
                            # Không cần check _shared_loop vì đang chạy trong matching thread
                            self._create_match_sync(entry1, entry2, bs)
                            # Nếu tạo thành công (không raise exception), đánh dấu để remove khỏi queue
                            successful_matches.append((i, j))
                            logger.info(
                                f"✅ Successfully created match for {entry1.user_id} vs {entry2.user_id} "
                                f"on {bs}x{bs} board"
                            )
                        except Exception as e:
                            logger.error(f"❌ Error creating match: {e}", exc_info=True)
                            # Không remove khỏi queue nếu tạo match fail
                            # Entries vẫn còn trong queue để có thể match lại
                    
                    # CHỈ remove entries khỏi queue sau khi match được tạo thành công
                    # Remove theo thứ tự ngược lại để giữ nguyên indices
                    # ĐỢI một chút để đảm bảo match đã được commit vào database
                    import time
                    time.sleep(0.2)  # Wait 200ms để đảm bảo match đã commit
                    
                    for i, j in sorted(successful_matches, reverse=True):
                        queue.pop(max(i, j))
                        queue.pop(min(i, j))
                        logger.info(f"Removed matched players from queue after match creation")
                    
                    # Remove timed out entries
                    now = datetime.now(timezone.utc)
                    timed_out = [
                        i for i, entry in enumerate(queue)
                        if (now - entry.joined_at).total_seconds() > QUEUE_TIMEOUT
                    ]
                    for i in sorted(timed_out, reverse=True):
                        entry = queue.pop(i)
                        logger.info(f"Removed timed out player {entry.user_id} from queue")
                    
                    # Remove empty queues
                    if not queue:
                        del MatchmakingService._shared_queue[board_size]
        except Exception as e:
            logger.error(f"Error in _try_match_players: {e}", exc_info=True)
            # Không re-raise để matching loop tiếp tục chạy
    
    async def _create_match(self, entry1: QueueEntry, entry2: QueueEntry, board_size: int):
        """Tạo match cho 2 players đã được match."""
        try:
            # Run in thread pool để tránh blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._create_match_sync, entry1, entry2, board_size)
        except asyncio.CancelledError:
            logger.warning("Match creation cancelled")
        except Exception as e:
            logger.error(f"Error creating match: {e}", exc_info=True)
    
    def _create_match_sync(self, entry1: QueueEntry, entry2: QueueEntry, board_size: int):
        """Tạo match synchronously (chạy trong thread pool)."""
        try:
            logger.info(
                f"Creating match: {entry1.user_id} (ELO {entry1.elo_rating}) vs "
                f"{entry2.user_id} (ELO {entry2.elo_rating}) on {board_size}x{board_size}"
            )
            
            # Validate board_size trước khi tạo match
            if entry1.board_size != board_size or entry2.board_size != board_size:
                logger.error(
                    f"Cannot create match - board size validation failed: "
                    f"entry1.board_size={entry1.board_size}, "
                    f"entry2.board_size={entry2.board_size}, "
                    f"expected={board_size}"
                )
                return
            
            # Get user objects với error handling
            user1 = self.db.get(user_model.User, entry1.user_id)
            user2 = self.db.get(user_model.User, entry2.user_id)
            
            if not user1:
                logger.error(f"User1 not found: {entry1.user_id}")
                return
            if not user2:
                logger.error(f"User2 not found: {entry2.user_id}")
                return
            
            # Determine black/white based on ELO (higher ELO = black)
            if entry1.elo_rating >= entry2.elo_rating:
                black_user = user1
                white_user = user2
            else:
                black_user = user2
                white_user = user1
            
            # Kiểm tra xem đã có match giữa 2 user này chưa - nếu có thì xóa match cũ
            existing_matches = (
                self.db.query(match_model.Match)
                .filter(
                    (
                        (match_model.Match.black_player_id == black_user.id) &
                        (match_model.Match.white_player_id == white_user.id)
                    ) | (
                        (match_model.Match.black_player_id == white_user.id) &
                        (match_model.Match.white_player_id == black_user.id)
                    ),
                    match_model.Match.result.is_(None),
                    match_model.Match.finished_at.is_(None)
                )
                .all()
            )
            
            if existing_matches:
                # Xóa tất cả matches cũ giữa 2 người chơi này
                for old_match in existing_matches:
                    logger.info(
                        f"🗑️ Deleting old match {old_match.id} between {black_user.id} and {white_user.id} "
                        f"(black_ready={old_match.black_ready}, white_ready={old_match.white_ready})"
                    )
                    self.db.delete(old_match)
                self.db.commit()
                logger.info(f"✅ Deleted {len(existing_matches)} old match(es) before creating new match")
            
            # Tạo mã bàn 6 ký tự duy nhất cho matchmaking match
            import random
            import string
            from uuid import uuid4
            max_attempts = 10
            room_code = None
            for _ in range(max_attempts):
                code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                # Kiểm tra mã đã tồn tại chưa - chỉ check các match đang active (chưa kết thúc)
                existing = self.db.query(match_model.Match).filter(
                    match_model.Match.room_code == code,
                    match_model.Match.finished_at.is_(None),  # Chưa kết thúc
                    match_model.Match.result.is_(None)  # Chưa có kết quả
                ).first()
                if not existing:
                    room_code = code
                    break
            
            if not room_code:
                # Fallback nếu không tạo được mã duy nhất
                room_code = uuid4().hex[:6].upper()
                logger.warning(f"Could not generate unique room code for matchmaking, using fallback: {room_code}")
            
            # Xác định thời gian cho mỗi người chơi dựa trên kích thước bàn cờ
            # 9x9  -> 10 phút
            # 13x13 -> 20 phút
            # 19x19 -> 30 phút
            if board_size == 9:
                time_control_minutes = 10
            elif board_size == 13:
                time_control_minutes = 20
            elif board_size == 19:
                time_control_minutes = 30
            else:
                # Fallback an toàn nếu sau này có board_size khác
                time_control_minutes = 10

            # Create match với board_size đã validate, room_code và thời gian phù hợp
            match = match_model.Match(
                black_player_id=black_user.id,
                white_player_id=white_user.id,
                board_size=board_size,  # Sử dụng board_size từ parameter (đã validate)
                room_code=room_code,  # Thêm room_code cho matchmaking matches
                time_control_minutes=time_control_minutes,
                black_time_remaining_seconds=time_control_minutes * 60,
                white_time_remaining_seconds=time_control_minutes * 60,
                last_move_at=datetime.now(timezone.utc),
                black_ready=False,  # Chưa ready - cần cả 2 người chơi xác nhận
                white_ready=False,  # Chưa ready - cần cả 2 người chơi xác nhận
            )
            self.db.add(match)
            self.db.flush()  # Flush để có ID ngay lập tức
            self.db.commit()  # Commit để match có thể query được
            self.db.refresh(match)  # Refresh để có đầy đủ thông tin
            
            # Đảm bảo match được flush vào database ngay lập tức
            # để cả 2 người chơi đều có thể tìm thấy match này
            logger.info(
                f"✅ Created match {match.id} with room_code {room_code} "
                f"for {black_user.id} (Black) vs {white_user.id} (White) "
                f"at {datetime.now(timezone.utc)}"
            )
            
            # Verify match exists in database
            verify_match = self.db.query(match_model.Match).filter(
                match_model.Match.id == match.id
            ).first()
            if verify_match:
                logger.info(f"✅ Verified match {match.id} exists in database")
            else:
                logger.error(f"❌ Match {match.id} not found in database after creation!")
            
            logger.info(
                f"✅ Successfully created match {match.id} for players "
                f"{black_user.id} (Black, ELO {black_user.elo_rating}) and "
                f"{white_user.id} (White, ELO {white_user.elo_rating}) "
                f"on {board_size}x{board_size} board with room_code {room_code}"
            )
            
            # KHÔNG remove users khỏi queue ở đây
            # Việc remove sẽ được thực hiện trong _try_match_players sau khi match được tạo thành công
            # Điều này đảm bảo rằng nếu có lỗi, users vẫn còn trong queue
            
        except Exception as e:
            logger.error(f"❌ Error creating match: {e}", exc_info=True)
            self.db.rollback()
            # Re-raise để caller biết có lỗi
            raise
    
    def join_queue(self, user_id: str, elo_rating: int, board_size: int) -> bool:
        """Thêm player vào queue.
        
        Args:
            user_id: User ID
            elo_rating: ELO rating của user
            board_size: Kích thước bàn cờ (9, 13, 19)
        
        Returns:
            True nếu join thành công, False nếu đã có trong queue
        """
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        logger.info(f"🔄 [JOIN_QUEUE_SERVICE] join_queue() called for user {user_id}, board_size={board_size}")
        print(f"🔄 [JOIN_QUEUE_SERVICE] join_queue() called for user {user_id}, board_size={board_size}")
        
        try:
            logger.info(f"🔒 [JOIN_QUEUE_SERVICE] Acquiring lock...")
            print(f"🔒 [JOIN_QUEUE_SERVICE] Acquiring lock...")
            with MatchmakingService._shared_lock:
                lock_time = time.time() - start_time
                logger.info(f"✅ [JOIN_QUEUE_SERVICE] Lock acquired in {lock_time:.3f}s")
                print(f"✅ [JOIN_QUEUE_SERVICE] Lock acquired in {lock_time:.3f}s")
                lock_time = time.time() - start_time
                if lock_time > 0.1:
                    logger.warning(f"⏱️ [JOIN_QUEUE] Waited {lock_time:.3f}s for lock")
                
                # Check if already in queue
                check_start = time.time()
                logger.info(f"🔍 [JOIN_QUEUE_SERVICE] Checking if user already in queue...")
                print(f"🔍 [JOIN_QUEUE_SERVICE] Checking if user already in queue...")
                for queue in MatchmakingService._shared_queue.values():
                    if any(entry.user_id == user_id for entry in queue):
                        logger.info(f"⚠️ [JOIN_QUEUE_SERVICE] User {user_id} already in queue")
                        print(f"⚠️ [JOIN_QUEUE_SERVICE] User {user_id} already in queue")
                        return False
                check_time = time.time() - check_start
                logger.info(f"✅ [JOIN_QUEUE_SERVICE] Queue check completed in {check_time:.3f}s")
                print(f"✅ [JOIN_QUEUE_SERVICE] Queue check completed in {check_time:.3f}s")
                if check_time > 0.01:
                    logger.warning(f"⏱️ [JOIN_QUEUE_SERVICE] Queue check took {check_time:.3f}s")
                
                # Add to queue
                logger.info(f"➕ [JOIN_QUEUE_SERVICE] Adding user to queue...")
                print(f"➕ [JOIN_QUEUE_SERVICE] Adding user to queue...")
                if board_size not in MatchmakingService._shared_queue:
                    MatchmakingService._shared_queue[board_size] = []
                    logger.info(f"📝 [JOIN_QUEUE_SERVICE] Created new queue for board_size={board_size}")
                    print(f"📝 [JOIN_QUEUE_SERVICE] Created new queue for board_size={board_size}")
                
                entry = QueueEntry(
                    user_id=user_id,
                    elo_rating=elo_rating,
                    board_size=board_size,
                    joined_at=datetime.now(timezone.utc)
                )
                MatchmakingService._shared_queue[board_size].append(entry)
                
                queue_size = len(MatchmakingService._shared_queue[board_size])
                logger.info(f"✅ [JOIN_QUEUE_SERVICE] Player {user_id} joined queue for {board_size}x{board_size} board (ELO: {elo_rating}, Queue size: {queue_size})")
                print(f"✅ [JOIN_QUEUE_SERVICE] Player {user_id} joined queue (Queue size: {queue_size})")
                
                # Check if matching task needs to be started (while still holding lock)
                should_start_matching = not MatchmakingService._shared_running
                logger.info(f"🚀 [JOIN_QUEUE_SERVICE] Should start matching task: {should_start_matching}")
                print(f"🚀 [JOIN_QUEUE_SERVICE] Should start matching task: {should_start_matching}")
            
            # Start matching task AFTER releasing lock to avoid deadlock
            if should_start_matching:
                logger.info(f"🚀 [JOIN_QUEUE_SERVICE] Starting matching task (outside lock)...")
                print(f"🚀 [JOIN_QUEUE_SERVICE] Starting matching task (outside lock)...")
                try:
                    self.start_matching_task()
                    logger.info(f"✅ [JOIN_QUEUE_SERVICE] Matching task started")
                    print(f"✅ [JOIN_QUEUE_SERVICE] Matching task started")
                except Exception as e:
                    logger.error(f"❌ [JOIN_QUEUE_SERVICE] Error starting matching task: {e}", exc_info=True)
                    print(f"❌ [JOIN_QUEUE_SERVICE] Error starting matching task: {e}")
                    # Không fail request nếu start matching task lỗi - user vẫn đã join queue
                    # Chỉ log warning
            else:
                logger.info(f"ℹ️ [JOIN_QUEUE_SERVICE] Matching task already running")
                print(f"ℹ️ [JOIN_QUEUE_SERVICE] Matching task already running")
            
            total_time = time.time() - start_time
            logger.info(f"✅ [JOIN_QUEUE_SERVICE] join_queue() completed in {total_time:.3f}s")
            print(f"✅ [JOIN_QUEUE_SERVICE] join_queue() completed in {total_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"❌ [JOIN_QUEUE] Error in join_queue: {e}", exc_info=True)
            raise
    
    def leave_queue(self, user_id: str) -> bool:
        """Xóa player khỏi queue.
        
        Returns:
            True nếu leave thành công, False nếu không có trong queue
        """
        with MatchmakingService._shared_lock:
            for board_size, queue in list(MatchmakingService._shared_queue.items()):
                for i, entry in enumerate(queue):
                    if entry.user_id == user_id:
                        queue.pop(i)
                        logger.info(f"Player {user_id} left queue")
                        # Remove empty queues
                        if not queue:
                            del MatchmakingService._shared_queue[board_size]
                        return True
        return False
    
    def get_queue_status(self, user_id: str) -> Optional[Dict]:
        """Lấy trạng thái queue của user.
        
        Returns:
            Dict với thông tin queue hoặc None nếu không có trong queue
        """
        with MatchmakingService._shared_lock:
            for board_size, queue in MatchmakingService._shared_queue.items():
                for entry in queue:
                    if entry.user_id == user_id:
                        elapsed = (datetime.now(timezone.utc) - entry.joined_at).total_seconds()
                        return {
                            "in_queue": True,
                            "board_size": board_size,
                            "elo_rating": entry.elo_rating,
                            "wait_time": int(elapsed),
                            "queue_size": len(queue),  # Số người trong queue (bao gồm cả user hiện tại)
                            "elo_range": entry.elo_range,
                        }
        return None
    
    def get_match_for_user(self, user_id: str) -> Optional[match_model.Match]:
        """Lấy match mới được tạo cho user (nếu có).
        
        Returns:
            Match object nếu có match mới, None nếu chưa có.
        """
        # Query database để tìm match mới được tạo cho user
        # Match phải có user là black_player hoặc white_player
        # Match phải chưa kết thúc (result = None, finished_at = None)
        # Match phải được tạo gần đây (trong vòng 30 phút) để tránh match cũ
        # HOẶC match chưa có cả 2 người ready (đang chờ ready)
        from datetime import timedelta
        
        try:
            # Tăng thời gian lên 1 giờ để đảm bảo tìm thấy match
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            
            # Query với điều kiện linh hoạt hơn
            matches = (
                self.db.query(match_model.Match)
                .filter(
                    (
                        (match_model.Match.black_player_id == user_id)
                        | (match_model.Match.white_player_id == user_id)
                    )
                    & (match_model.Match.result.is_(None))
                    & (match_model.Match.finished_at.is_(None))
                    & (match_model.Match.ai_level.is_(None))  # Chỉ PvP matches
                    & (
                        # Match mới được tạo (trong 1 giờ) HOẶC chưa có cả 2 ready
                        (match_model.Match.started_at >= one_hour_ago)
                        | (
                            (match_model.Match.black_ready == False)
                            | (match_model.Match.white_ready == False)
                        )
                    )
                )
                .order_by(match_model.Match.started_at.desc())
                .all()
            )
            
            if matches:
                # Lấy match mới nhất
                match = matches[0]
                logger.info(
                    f"✅ Found match {match.id} for user {user_id} "
                    f"(black={match.black_player_id}, white={match.white_player_id}, "
                    f"room_code={match.room_code}, "
                    f"black_ready={match.black_ready}, white_ready={match.white_ready}, "
                    f"started_at={match.started_at})"
                )
                if len(matches) > 1:
                    logger.warning(
                        f"⚠️ Found {len(matches)} matches for user {user_id}, "
                        f"returning the most recent one: {match.id}"
                    )
                return match
            else:
                logger.debug(f"ℹ️ No match found for user {user_id}")
                return None
        except Exception as e:
            logger.error(f"❌ Error getting match for user {user_id}: {e}", exc_info=True)
            return None
    
    def get_queue_stats(self) -> Dict:
        """Lấy thống kê queue (cho admin/monitoring)."""
        with MatchmakingService._shared_lock:
            total_players = sum(len(queue) for queue in MatchmakingService._shared_queue.values())
            return {
                "total_players": total_players,
                "by_board_size": {
                    board_size: len(queue)
                    for board_size, queue in MatchmakingService._shared_queue.items()
                },
                "running": MatchmakingService._shared_running,
            }
