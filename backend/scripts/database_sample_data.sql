-- ============================================================
-- GoGame Database Sample Data Script
-- ============================================================
-- Script SQL để insert dữ liệu mẫu cho testing
-- 
-- Usage:
--   psql -U postgres -d gogame -f scripts/database_sample_data.sql
-- ============================================================

\c gogame

-- ============================================================
-- 1. INSERT SAMPLE USERS
-- ============================================================

-- Xóa dữ liệu cũ (tùy chọn - comment nếu muốn giữ dữ liệu cũ)
-- TRUNCATE TABLE users CASCADE;

-- Insert sample users
INSERT INTO users (id, username, email, password_hash, elo_rating, coins, display_name, created_at) VALUES
    ('00000000-0000-0000-0000-000000000001', 'testuser1', 'test1@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyY5Y5Y5Y5Y5', 1500, 100, 'Test User 1', NOW()),
    ('00000000-0000-0000-0000-000000000002', 'testuser2', 'test2@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyY5Y5Y5Y5', 1600, 200, 'Test User 2', NOW()),
    ('00000000-0000-0000-0000-000000000003', 'testuser3', 'test3@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyY5Y5Y5Y5', 1400, 50, 'Test User 3', NOW())
ON CONFLICT (id) DO NOTHING;

\echo '✅ Đã insert sample users'

-- ============================================================
-- 2. INSERT SAMPLE MATCHES
-- ============================================================

-- Insert sample matches
INSERT INTO matches (id, black_player_id, white_player_id, board_size, result, started_at, finished_at) VALUES
    ('10000000-0000-0000-0000-000000000001'::UUID, 
     '00000000-0000-0000-0000-000000000001'::UUID, 
     '00000000-0000-0000-0000-000000000002'::UUID, 
     9, 'black_wins', NOW() - INTERVAL '1 hour', NOW() - INTERVAL '30 minutes'),
    ('10000000-0000-0000-0000-000000000002'::UUID, 
     '00000000-0000-0000-0000-000000000002'::UUID, 
     NULL, 
     13, 'white_wins', NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour'),
    ('10000000-0000-0000-0000-000000000003'::UUID, 
     NULL, 
     '00000000-0000-0000-0000-000000000003'::UUID, 
     19, NULL, NOW() - INTERVAL '10 minutes', NULL)
ON CONFLICT (id) DO NOTHING;

\echo '✅ Đã insert sample matches'

-- ============================================================
-- 3. INSERT SAMPLE COIN TRANSACTIONS
-- ============================================================

INSERT INTO coin_transactions (id, user_id, amount, type, source, created_at) VALUES
    ('20000000-0000-0000-0000-000000000001'::UUID, '00000000-0000-0000-0000-000000000001'::UUID, 50, 'earn', 'match_win', NOW() - INTERVAL '1 hour'),
    ('20000000-0000-0000-0000-000000000002'::UUID, '00000000-0000-0000-0000-000000000001'::UUID, -10, 'spend', 'premium_analysis', NOW() - INTERVAL '30 minutes'),
    ('20000000-0000-0000-0000-000000000003'::UUID, '00000000-0000-0000-0000-000000000002'::UUID, 100, 'bonus', 'welcome', NOW() - INTERVAL '2 hours')
ON CONFLICT (id) DO NOTHING;

\echo '✅ Đã insert sample coin transactions'

-- ============================================================
-- 4. HIỂN THỊ THỐNG KÊ
-- ============================================================

\echo ''
\echo '============================================================'
\echo '✅ Sample data đã được insert thành công!'
\echo '============================================================'
\echo ''
\echo '📊 Thống kê:'
\echo ''
\echo 'Users:'
SELECT COUNT(*) as total_users FROM users;
SELECT username, email, elo_rating, coins FROM users ORDER BY created_at LIMIT 5;
\echo ''
\echo 'Matches:'
SELECT COUNT(*) as total_matches FROM matches;
SELECT id, board_size, result, started_at FROM matches ORDER BY started_at DESC LIMIT 5;
\echo ''
\echo 'Coin Transactions:'
SELECT COUNT(*) as total_transactions FROM coin_transactions;
SELECT type, SUM(amount) as total_amount FROM coin_transactions GROUP BY type;
\echo ''

