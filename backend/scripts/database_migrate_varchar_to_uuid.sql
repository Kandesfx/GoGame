-- ============================================================
-- Migration Script: VARCHAR(36) to UUID
-- ============================================================
-- Script này chuyển đổi các cột ID từ VARCHAR(36) sang UUID type
-- Chạy script này nếu database đã được tạo với VARCHAR(36)
--
-- ⚠️ CẢNH BÁO: Backup database trước khi chạy!
--
-- Usage:
--   psql -U postgres -d gogame -f scripts/database_migrate_varchar_to_uuid.sql
-- ============================================================

\c gogame

BEGIN;

\echo '🔄 Đang chuyển đổi các cột ID từ VARCHAR(36) sang UUID...'
\echo ''

-- 1. Users table
\echo '1. Chuyển đổi bảng users...'
ALTER TABLE users 
    ALTER COLUMN id TYPE UUID USING id::UUID;

\echo '   ✅ users.id đã được chuyển đổi'
\echo ''

-- 2. Matches table
\echo '2. Chuyển đổi bảng matches...'
-- Xóa foreign keys tạm thời
ALTER TABLE matches DROP CONSTRAINT IF EXISTS fk_matches_black_player;
ALTER TABLE matches DROP CONSTRAINT IF EXISTS fk_matches_white_player;

-- Chuyển đổi các cột
ALTER TABLE matches 
    ALTER COLUMN id TYPE UUID USING id::UUID,
    ALTER COLUMN black_player_id TYPE UUID USING NULLIF(black_player_id, '')::UUID,
    ALTER COLUMN white_player_id TYPE UUID USING NULLIF(white_player_id, '')::UUID;

-- Tạo lại foreign keys
ALTER TABLE matches 
    ADD CONSTRAINT fk_matches_black_player 
    FOREIGN KEY (black_player_id) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE matches 
    ADD CONSTRAINT fk_matches_white_player 
    FOREIGN KEY (white_player_id) REFERENCES users(id) ON DELETE SET NULL;

\echo '   ✅ matches.id, black_player_id, white_player_id đã được chuyển đổi'
\echo ''

-- 3. Refresh tokens table
\echo '3. Chuyển đổi bảng refresh_tokens...'
ALTER TABLE refresh_tokens DROP CONSTRAINT IF EXISTS fk_refresh_tokens_user;

ALTER TABLE refresh_tokens 
    ALTER COLUMN id TYPE UUID USING id::UUID,
    ALTER COLUMN user_id TYPE UUID USING user_id::UUID;

ALTER TABLE refresh_tokens 
    ADD CONSTRAINT fk_refresh_tokens_user 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

\echo '   ✅ refresh_tokens.id, user_id đã được chuyển đổi'
\echo ''

-- 4. Coin transactions table
\echo '4. Chuyển đổi bảng coin_transactions...'
ALTER TABLE coin_transactions DROP CONSTRAINT IF EXISTS fk_coin_transactions_user;

ALTER TABLE coin_transactions 
    ALTER COLUMN id TYPE UUID USING id::UUID,
    ALTER COLUMN user_id TYPE UUID USING user_id::UUID;

ALTER TABLE coin_transactions 
    ADD CONSTRAINT fk_coin_transactions_user 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

\echo '   ✅ coin_transactions.id, user_id đã được chuyển đổi'
\echo ''

-- 5. Premium requests table
\echo '5. Chuyển đổi bảng premium_requests...'
ALTER TABLE premium_requests DROP CONSTRAINT IF EXISTS fk_premium_requests_user;
ALTER TABLE premium_requests DROP CONSTRAINT IF EXISTS fk_premium_requests_match;

ALTER TABLE premium_requests 
    ALTER COLUMN id TYPE UUID USING id::UUID,
    ALTER COLUMN user_id TYPE UUID USING user_id::UUID,
    ALTER COLUMN match_id TYPE UUID USING match_id::UUID;

ALTER TABLE premium_requests 
    ADD CONSTRAINT fk_premium_requests_user 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE premium_requests 
    ADD CONSTRAINT fk_premium_requests_match 
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE;

\echo '   ✅ premium_requests.id, user_id, match_id đã được chuyển đổi'
\echo ''

COMMIT;

\echo ''
\echo '============================================================'
\echo '✅ Migration hoàn tất!'
\echo '============================================================'
\echo ''
\echo '📊 Kiểm tra kiểu dữ liệu:'
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public' 
    AND column_name IN ('id', 'user_id', 'match_id', 'black_player_id', 'white_player_id')
ORDER BY table_name, column_name;
\echo ''

