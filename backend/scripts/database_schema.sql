-- ============================================================
-- GoGame Database Schema Script
-- ============================================================
-- Script SQL để tạo toàn bộ database schema cho GoGame
-- Chạy script này với quyền superuser (postgres)
--
-- Usage:
--   psql -U postgres -f scripts/database_schema.sql
--   hoặc
--   psql -U postgres
--   \i scripts/database_schema.sql
-- ============================================================

-- ============================================================
-- 1. TẠO DATABASE
-- ============================================================

-- Tạo database (chạy từ database postgres)

-- Xóa database cũ nếu tồn tại (CẨN THẬN: Sẽ mất dữ liệu!)
-- DROP DATABASE IF EXISTS gogame;

-- Tạo database mới
CREATE DATABASE gogame
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Kết nối đến database gogame

-- ============================================================
-- 2. TẠO EXTENSIONS
-- ============================================================

-- Extension cho UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 3. TẠO BẢNG USERS
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(32) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    elo_rating INTEGER DEFAULT 1500 NOT NULL,
    coins INTEGER DEFAULT 0 NOT NULL,
    display_name VARCHAR(64),
    avatar_url VARCHAR(255),
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Indexes cho bảng users
CREATE INDEX IF NOT EXISTS ix_users_username ON users(username);
CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
CREATE INDEX IF NOT EXISTS ix_users_elo_rating ON users(elo_rating);

-- Comments
COMMENT ON TABLE users IS 'Bảng người dùng';
COMMENT ON COLUMN users.id IS 'UUID của user';
COMMENT ON COLUMN users.username IS 'Tên đăng nhập (unique)';
COMMENT ON COLUMN users.email IS 'Email (unique)';
COMMENT ON COLUMN users.elo_rating IS 'Điểm ELO rating (mặc định 1500)';
COMMENT ON COLUMN users.coins IS 'Số coins của user';
COMMENT ON COLUMN users.preferences IS 'Preferences dạng JSON';

-- ============================================================
-- 4. TẠO BẢNG MATCHES
-- ============================================================

CREATE TABLE IF NOT EXISTS matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    black_player_id UUID,
    white_player_id UUID,
    ai_level INTEGER,
    board_size INTEGER DEFAULT 9 NOT NULL,
    result VARCHAR(32),
    room_code VARCHAR(6) UNIQUE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    finished_at TIMESTAMP WITH TIME ZONE,
    sgf_id VARCHAR(64),
    premium_analysis_id VARCHAR(64),
    -- Time control cho PvP matches
    time_control_minutes INTEGER,
    black_time_remaining_seconds INTEGER,
    white_time_remaining_seconds INTEGER,
    last_move_at TIMESTAMP WITH TIME ZONE,
    -- ELO changes (chỉ cho PvP matches)
    black_elo_change INTEGER,
    white_elo_change INTEGER,
    -- Ready status cho matchmaking (chỉ cho PvP matches)
    black_ready BOOLEAN DEFAULT FALSE NOT NULL,
    white_ready BOOLEAN DEFAULT FALSE NOT NULL,
    -- Foreign keys
    CONSTRAINT fk_matches_black_player FOREIGN KEY (black_player_id) 
        REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_matches_white_player FOREIGN KEY (white_player_id) 
        REFERENCES users(id) ON DELETE SET NULL
);

-- Indexes cho bảng matches
CREATE INDEX IF NOT EXISTS ix_matches_room_code ON matches(room_code) WHERE room_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_matches_black_player_id ON matches(black_player_id) WHERE black_player_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_matches_white_player_id ON matches(white_player_id) WHERE white_player_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_matches_started_at ON matches(started_at);
CREATE INDEX IF NOT EXISTS ix_matches_finished_at ON matches(finished_at) WHERE finished_at IS NOT NULL;

-- Comments
COMMENT ON TABLE matches IS 'Bảng trận đấu';
COMMENT ON COLUMN matches.id IS 'UUID của match';
COMMENT ON COLUMN matches.black_player_id IS 'ID của người chơi đen (NULL nếu là AI)';
COMMENT ON COLUMN matches.white_player_id IS 'ID của người chơi trắng (NULL nếu là AI)';
COMMENT ON COLUMN matches.ai_level IS 'Mức độ AI (1-4, NULL nếu là PvP)';
COMMENT ON COLUMN matches.board_size IS 'Kích thước bàn cờ (9, 13, hoặc 19)';
COMMENT ON COLUMN matches.result IS 'Kết quả trận đấu';
COMMENT ON COLUMN matches.room_code IS 'Mã phòng 6 ký tự (cho PvP)';
COMMENT ON COLUMN matches.black_ready IS 'Black player đã sẵn sàng';
COMMENT ON COLUMN matches.white_ready IS 'White player đã sẵn sàng';

-- ============================================================
-- 5. TẠO BẢNG REFRESH_TOKENS
-- ============================================================

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    token TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked BOOLEAN DEFAULT FALSE NOT NULL,
    -- Foreign key
    CONSTRAINT fk_refresh_tokens_user FOREIGN KEY (user_id) 
        REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes cho bảng refresh_tokens
CREATE INDEX IF NOT EXISTS ix_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS ix_refresh_tokens_token ON refresh_tokens(token);
CREATE INDEX IF NOT EXISTS ix_refresh_tokens_expires_at ON refresh_tokens(expires_at);

-- Comments
COMMENT ON TABLE refresh_tokens IS 'Bảng refresh tokens cho JWT authentication';
COMMENT ON COLUMN refresh_tokens.token IS 'Refresh token (hashed)';
COMMENT ON COLUMN refresh_tokens.expires_at IS 'Thời gian hết hạn';
COMMENT ON COLUMN refresh_tokens.revoked IS 'Token đã bị thu hồi';

-- ============================================================
-- 6. TẠO BẢNG COIN_TRANSACTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS coin_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    amount INTEGER NOT NULL,
    type VARCHAR(32) NOT NULL,
    source VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    -- Foreign key
    CONSTRAINT fk_coin_transactions_user FOREIGN KEY (user_id) 
        REFERENCES users(id) ON DELETE CASCADE,
    -- Check constraint
    CONSTRAINT chk_coin_transactions_amount CHECK (amount != 0)
);

-- Indexes cho bảng coin_transactions
CREATE INDEX IF NOT EXISTS ix_coin_transactions_user_id ON coin_transactions(user_id);
CREATE INDEX IF NOT EXISTS ix_coin_transactions_created_at ON coin_transactions(created_at);
CREATE INDEX IF NOT EXISTS ix_coin_transactions_type ON coin_transactions(type);

-- Comments
COMMENT ON TABLE coin_transactions IS 'Bảng lịch sử giao dịch coins';
COMMENT ON COLUMN coin_transactions.amount IS 'Số coins (dương = thêm, âm = trừ)';
COMMENT ON COLUMN coin_transactions.type IS 'Loại giao dịch (earn, spend, bonus, etc.)';
COMMENT ON COLUMN coin_transactions.source IS 'Nguồn giao dịch';

-- ============================================================
-- 7. TẠO BẢNG PREMIUM_REQUESTS
-- ============================================================

CREATE TABLE IF NOT EXISTS premium_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID NOT NULL,
    feature VARCHAR(32) NOT NULL,
    cost INTEGER NOT NULL,
    status VARCHAR(32) DEFAULT 'pending' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    -- Foreign keys
    CONSTRAINT fk_premium_requests_user FOREIGN KEY (user_id) 
        REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_premium_requests_match FOREIGN KEY (match_id) 
        REFERENCES matches(id) ON DELETE CASCADE
);

-- Indexes cho bảng premium_requests
CREATE INDEX IF NOT EXISTS ix_premium_requests_user_id ON premium_requests(user_id);
CREATE INDEX IF NOT EXISTS ix_premium_requests_match_id ON premium_requests(match_id);
CREATE INDEX IF NOT EXISTS ix_premium_requests_status ON premium_requests(status);
CREATE INDEX IF NOT EXISTS ix_premium_requests_created_at ON premium_requests(created_at);

-- Comments
COMMENT ON TABLE premium_requests IS 'Bảng yêu cầu phân tích premium';
COMMENT ON COLUMN premium_requests.feature IS 'Tính năng premium (analysis, review, etc.)';
COMMENT ON COLUMN premium_requests.cost IS 'Chi phí (coins)';
COMMENT ON COLUMN premium_requests.status IS 'Trạng thái (pending, processing, completed, failed)';

-- ============================================================
-- 8. TẠO BẢNG PREMIUM_SUBSCRIPTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS premium_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL,
    plan VARCHAR(32) NOT NULL,
    status VARCHAR(32) DEFAULT 'active' NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    -- Foreign key
    CONSTRAINT fk_premium_subscriptions_user FOREIGN KEY (user_id) 
        REFERENCES users(id) ON DELETE CASCADE,
    -- Check constraints
    CONSTRAINT chk_premium_subscriptions_plan CHECK (plan IN ('monthly', 'yearly')),
    CONSTRAINT chk_premium_subscriptions_status CHECK (status IN ('active', 'expired', 'cancelled'))
);

-- Indexes cho bảng premium_subscriptions
CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_user_id ON premium_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_status ON premium_subscriptions(status);
CREATE INDEX IF NOT EXISTS ix_premium_subscriptions_expires_at ON premium_subscriptions(expires_at);

-- Comments
COMMENT ON TABLE premium_subscriptions IS 'Bảng premium subscriptions của users';
COMMENT ON COLUMN premium_subscriptions.user_id IS 'ID của user (unique - mỗi user chỉ có 1 subscription)';
COMMENT ON COLUMN premium_subscriptions.plan IS 'Gói subscription (monthly hoặc yearly)';
COMMENT ON COLUMN premium_subscriptions.status IS 'Trạng thái (active, expired, cancelled)';
COMMENT ON COLUMN premium_subscriptions.started_at IS 'Thời gian bắt đầu subscription';
COMMENT ON COLUMN premium_subscriptions.expires_at IS 'Thời gian hết hạn subscription';
COMMENT ON COLUMN premium_subscriptions.cancelled_at IS 'Thời gian hủy subscription (nếu có)';

-- ============================================================
-- 9. TẠO BẢNG ALEMBIC_VERSION (cho migrations)
-- ============================================================

CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

COMMENT ON TABLE alembic_version IS 'Bảng tracking Alembic migrations';

-- ============================================================
-- 10. GRANT PERMISSIONS
-- ============================================================

-- Cấp quyền cho user postgres (nếu cần)
GRANT ALL PRIVILEGES ON DATABASE gogame TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Nếu muốn tạo user riêng (tùy chọn)
-- CREATE USER gogame_user WITH PASSWORD 'your_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE gogame TO gogame_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gogame_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gogame_user;



