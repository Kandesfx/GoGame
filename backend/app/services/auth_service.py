"""Nghiệp vụ xác thực và JWT."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from uuid import UUID, uuid4

import jwt
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..config import Settings
from ..models.sql import refresh_token as refresh_token_model
from ..models.sql import user as user_model
from ..utils import security


class AuthService:
    """Xử lý đăng ký, đăng nhập, phát hành JWT."""

    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def register(self, username: str, email: str, password: str) -> user_model.User:
        # Check if username exists
        existing_user = self.db.query(user_model.User).filter(
            (user_model.User.username == username) | (user_model.User.email == email)
        ).first()
        
        if existing_user:
            if existing_user.username == username:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Username '{username}' already exists"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Email '{email}' already exists"
                )

        try:
            hashed_pw = security.hash_password(password)
            user = user_model.User(username=username, email=email, password_hash=hashed_pw)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except Exception as e:
            self.db.rollback()
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Registration failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Registration failed: {str(e)}"
            )

    def authenticate(self, username_or_email: str, password: str) -> user_model.User:
        user = (
            self.db.query(user_model.User)
            .filter(
                (user_model.User.username == username_or_email) | (user_model.User.email == username_or_email)
            )
            .first()
        )
        if not user or not security.verify_password(password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        user.last_login = datetime.now(tz=timezone.utc)
        self.db.commit()
        return user

    def issue_tokens(self, user: user_model.User) -> Dict[str, str]:
        refresh_id = str(uuid4())
        refresh_token = refresh_token_model.RefreshToken(
            id=refresh_id,
            user_id=user.id,
            token="",  # placeholder, cập nhật sau
            expires_at=datetime.now(tz=timezone.utc)
            + self._delta_from_days(self.settings.refresh_token_exp_days),
        )

        refresh_jwt = security.create_refresh_token(
            subject=user.id,
            settings=self.settings,
            token_id=refresh_id,
        )
        refresh_token.token = refresh_jwt

        access_jwt = security.create_access_token(subject=user.id, settings=self.settings)

        # lưu refresh token
        self.db.add(refresh_token)
        self.db.commit()

        return {"access_token": access_jwt, "refresh_token": refresh_jwt}

    def refresh(self, refresh_token: str) -> Dict[str, str]:
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            payload = security.decode_refresh_token(refresh_token, self.settings)
        except jwt.ExpiredSignatureError as exc:
            logger.warning(f"❌ [AUTH] Refresh token expired: {exc}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired") from exc
        except jwt.InvalidTokenError as exc:
            logger.warning(f"❌ [AUTH] Invalid refresh token format: {exc}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc
        except jwt.PyJWTError as exc:
            logger.warning(f"❌ [AUTH] JWT error during refresh: {exc}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc

        token_id = payload.get("jti")
        if not token_id:
            logger.warning(f"❌ [AUTH] Refresh token missing jti claim")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        db_token = self.db.query(refresh_token_model.RefreshToken).filter_by(id=token_id, revoked=False).first()
        if not db_token:
            logger.warning(f"❌ [AUTH] Refresh token not found in database: token_id={token_id}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token not found")
        
        if db_token.token != refresh_token:
            # Token đã được rotate bởi request khác - đây là hành vi bình thường (race condition)
            # Không phải lỗi nghiêm trọng, chỉ log ở mức INFO
            logger.info(f"ℹ️ [AUTH] Refresh token was rotated by another request: token_id={token_id}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked or rotated")

        # Extend refresh token expiration khi có activity (sliding session)
        # QUAN TRỌNG: Luôn extend khi có request (không có điều kiện thời gian tối thiểu)
        now = datetime.now(tz=timezone.utc)
        
        # Luôn extend thêm N ngày từ bây giờ (sliding window)
        new_expires_at = now + self._delta_from_days(self.settings.refresh_token_exp_days)
        db_token.expires_at = new_expires_at
        
        # Tạo refresh token mới với expiration mới
        new_refresh_jwt = security.create_refresh_token(
            subject=payload["sub"],
            settings=self.settings,
            token_id=token_id,
        )
        db_token.token = new_refresh_jwt
        self.db.commit()
        refresh_token = new_refresh_jwt

        access_jwt = security.create_access_token(subject=payload["sub"], settings=self.settings)
        return {"access_token": access_jwt, "refresh_token": refresh_token}

    def revoke_refresh_token(self, refresh_token: str) -> None:
        record = self.db.query(refresh_token_model.RefreshToken).filter_by(token=refresh_token).first()
        if record:
            record.revoked = True
            self.db.commit()

    def decode_access_token(self, token: str) -> Dict[str, str]:
        try:
            return security.decode_access_token(token, self.settings)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    def get_user(self, user_id: UUID) -> Optional[user_model.User]:
        """Lấy user từ database. Tối ưu để nhanh nhất có thể."""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        try:
            # Sử dụng db.get() - nhanh nhất vì dùng primary key
            user = self.db.get(user_model.User, str(user_id))
            elapsed = time.time() - start_time
            if elapsed > 0.05:  # Cảnh báo nếu > 50ms
                logger.warning(f"⏱️ [AUTH] get_user({user_id}) took {elapsed:.3f}s - DB may be slow")
            return user
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ [AUTH] get_user({user_id}) failed after {elapsed:.3f}s: {e}", exc_info=True)
            raise
    
    def extend_session_if_active(self, user_id: UUID) -> None:
        """Extend refresh token expiration nếu user đang active (có request).
        
        Sliding session: Mỗi khi có request, extend thêm N ngày từ bây giờ.
        Chỉ extend nếu refresh token còn hạn (chưa expired).
        KHÔNG có điều kiện thời gian tối thiểu - extend ngay khi có request.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        now = datetime.now(tz=timezone.utc)
        
        # Tìm refresh token còn hạn và chưa bị revoke của user
        query_start = time.time()
        active_token = (
            self.db.query(refresh_token_model.RefreshToken)
            .filter_by(user_id=str(user_id), revoked=False)
            .filter(refresh_token_model.RefreshToken.expires_at > now)
            .order_by(refresh_token_model.RefreshToken.expires_at.desc())
            .first()
        )
        query_time = time.time() - query_start
        if query_time > 0.1:
            logger.warning(f"⏱️ [AUTH] RefreshToken query took {query_time:.3f}s")
        
        if not active_token:
            # Không có token active - không làm gì (user sẽ phải login lại)
            return
        
        # QUAN TRỌNG: Luôn extend khi có request (không có điều kiện thời gian tối thiểu)
        # Extend thêm N ngày từ bây giờ (sliding window)
        new_expires_at = now + self._delta_from_days(self.settings.refresh_token_exp_days)
        active_token.expires_at = new_expires_at
        
        # Tạo refresh token mới với expiration mới
        token_start = time.time()
        new_refresh_jwt = security.create_refresh_token(
            subject=str(user_id),
            settings=self.settings,
            token_id=active_token.id,
        )
        token_time = time.time() - token_start
        if token_time > 0.1:
            logger.warning(f"⏱️ [AUTH] create_refresh_token took {token_time:.3f}s")
        
        active_token.token = new_refresh_jwt
        
        commit_start = time.time()
        self.db.commit()
        commit_time = time.time() - commit_start
        if commit_time > 0.1:
            logger.warning(f"⏱️ [AUTH] DB commit took {commit_time:.3f}s")
        
        total_time = time.time() - start_time
        if total_time > 0.1:
            logger.warning(f"⏱️ [AUTH] extend_session_if_active total took {total_time:.3f}s")
        else:
            logger.debug(f"Extended session for user {user_id}: new expiry = {new_expires_at}")

    @staticmethod
    def _delta_from_days(days: int) -> timedelta:
        return timedelta(days=days)

