-- ============================================================
-- GoGame Database Useful Queries
-- ============================================================
-- Các câu query hữu ích để kiểm tra và quản lý database
-- ============================================================

\c gogame

-- ============================================================
-- 1. THÔNG TIN DATABASE
-- ============================================================

\echo '============================================================'
\echo '📊 THÔNG TIN DATABASE'
\echo '============================================================'
\echo ''

-- Kích thước database
SELECT 
    pg_size_pretty(pg_database_size('gogame')) AS database_size;

-- Danh sách tất cả các bảng
SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size('public.' || table_name)) AS size
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
ORDER BY pg_total_relation_size('public.' || table_name) DESC;

-- ============================================================
-- 2. THỐNG KÊ USERS
-- ============================================================

\echo ''
\echo '============================================================'
\echo '👥 THỐNG KÊ USERS'
\echo '============================================================'
\echo ''

-- Tổng số users
SELECT COUNT(*) AS total_users FROM users;

-- Top users theo ELO
SELECT username, elo_rating, coins, created_at 
FROM users 
ORDER BY elo_rating DESC 
LIMIT 10;

-- Users mới nhất
SELECT username, email, elo_rating, created_at 
FROM users 
ORDER BY created_at DESC 
LIMIT 10;

-- Phân bố ELO
SELECT 
    CASE 
        WHEN elo_rating < 1200 THEN '< 1200'
        WHEN elo_rating < 1500 THEN '1200-1500'
        WHEN elo_rating < 1800 THEN '1500-1800'
        WHEN elo_rating < 2100 THEN '1800-2100'
        ELSE '>= 2100'
    END AS elo_range,
    COUNT(*) AS user_count
FROM users
GROUP BY 
    CASE 
        WHEN elo_rating < 1200 THEN '< 1200'
        WHEN elo_rating < 1500 THEN '1200-1500'
        WHEN elo_rating < 1800 THEN '1500-1800'
        WHEN elo_rating < 2100 THEN '1800-2100'
        ELSE '>= 2100'
    END
ORDER BY elo_range;

-- ============================================================
-- 3. THỐNG KÊ MATCHES
-- ============================================================

\echo ''
\echo '============================================================'
\echo '🎮 THỐNG KÊ MATCHES'
\echo '============================================================'
\echo ''

-- Tổng số matches
SELECT COUNT(*) AS total_matches FROM matches;

-- Matches theo loại
SELECT 
    CASE 
        WHEN black_player_id IS NOT NULL AND white_player_id IS NOT NULL THEN 'PvP'
        WHEN ai_level IS NOT NULL THEN 'vs AI'
        ELSE 'Unknown'
    END AS match_type,
    COUNT(*) AS count
FROM matches
GROUP BY 
    CASE 
        WHEN black_player_id IS NOT NULL AND white_player_id IS NOT NULL THEN 'PvP'
        WHEN ai_level IS NOT NULL THEN 'vs AI'
        ELSE 'Unknown'
    END;

-- Matches theo kích thước bàn cờ
SELECT board_size, COUNT(*) AS count 
FROM matches 
GROUP BY board_size 
ORDER BY board_size;

-- Matches đang chơi (chưa kết thúc)
SELECT COUNT(*) AS active_matches 
FROM matches 
WHERE finished_at IS NULL;

-- Matches gần đây
SELECT 
    id,
    board_size,
    result,
    started_at,
    finished_at,
    CASE 
        WHEN finished_at IS NOT NULL THEN 
            EXTRACT(EPOCH FROM (finished_at - started_at)) / 60
        ELSE NULL
    END AS duration_minutes
FROM matches 
ORDER BY started_at DESC 
LIMIT 10;

-- ============================================================
-- 4. THỐNG KÊ COINS
-- ============================================================

\echo ''
\echo '============================================================'
\echo '💰 THỐNG KÊ COINS'
\echo '============================================================'
\echo ''

-- Tổng coins của tất cả users
SELECT SUM(coins) AS total_coins FROM users;

-- Top users theo coins
SELECT username, coins, elo_rating 
FROM users 
ORDER BY coins DESC 
LIMIT 10;

-- Thống kê giao dịch
SELECT 
    type,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount
FROM coin_transactions
GROUP BY type
ORDER BY transaction_count DESC;

-- Giao dịch gần đây
SELECT 
    u.username,
    ct.amount,
    ct.type,
    ct.source,
    ct.created_at
FROM coin_transactions ct
JOIN users u ON ct.user_id = u.id
ORDER BY ct.created_at DESC
LIMIT 10;

-- ============================================================
-- 5. THỐNG KÊ PREMIUM REQUESTS
-- ============================================================

\echo ''
\echo '============================================================'
\echo '⭐ THỐNG KÊ PREMIUM REQUESTS'
\echo '============================================================'
\echo ''

-- Thống kê theo trạng thái
SELECT 
    status,
    COUNT(*) AS count
FROM premium_requests
GROUP BY status
ORDER BY count DESC;

-- Premium requests gần đây
SELECT 
    u.username,
    pr.feature,
    pr.cost,
    pr.status,
    pr.created_at,
    pr.completed_at
FROM premium_requests pr
JOIN users u ON pr.user_id = u.id
ORDER BY pr.created_at DESC
LIMIT 10;

-- ============================================================
-- 6. THỐNG KÊ PREMIUM SUBSCRIPTIONS
-- ============================================================

\echo ''
\echo '============================================================'
\echo '⭐ THỐNG KÊ PREMIUM SUBSCRIPTIONS'
\echo '============================================================'
\echo ''

-- Tổng số subscriptions
SELECT COUNT(*) AS total_subscriptions FROM premium_subscriptions;

-- Subscriptions theo trạng thái
SELECT 
    status,
    COUNT(*) AS count
FROM premium_subscriptions
GROUP BY status
ORDER BY count DESC;

-- Subscriptions theo plan
SELECT 
    plan,
    COUNT(*) AS count
FROM premium_subscriptions
GROUP BY plan
ORDER BY count DESC;

-- Active subscriptions
SELECT 
    u.username,
    ps.plan,
    ps.status,
    ps.started_at,
    ps.expires_at,
    CASE 
        WHEN ps.expires_at > NOW() THEN 
            EXTRACT(EPOCH FROM (ps.expires_at - NOW())) / 86400
        ELSE 0
    END AS days_remaining
FROM premium_subscriptions ps
JOIN users u ON ps.user_id = u.id
WHERE ps.status = 'active'
ORDER BY ps.expires_at ASC;

-- Subscriptions sắp hết hạn (trong 7 ngày)
SELECT 
    u.username,
    ps.plan,
    ps.expires_at,
    EXTRACT(EPOCH FROM (ps.expires_at - NOW())) / 86400 AS days_remaining
FROM premium_subscriptions ps
JOIN users u ON ps.user_id = u.id
WHERE ps.status = 'active'
  AND ps.expires_at > NOW()
  AND ps.expires_at <= NOW() + INTERVAL '7 days'
ORDER BY ps.expires_at ASC;

-- Subscriptions đã hết hạn nhưng chưa update status
SELECT 
    u.username,
    ps.plan,
    ps.status,
    ps.expires_at,
    NOW() - ps.expires_at AS expired_duration
FROM premium_subscriptions ps
JOIN users u ON ps.user_id = u.id
WHERE ps.status = 'active'
  AND ps.expires_at < NOW()
ORDER BY ps.expires_at DESC;

-- ============================================================
-- 7. MAINTENANCE QUERIES
-- ============================================================

\echo ''
\echo '============================================================'
\echo '🔧 MAINTENANCE QUERIES'
\echo '============================================================'
\echo ''

-- Xóa refresh tokens đã hết hạn
-- DELETE FROM refresh_tokens WHERE expires_at < NOW();

-- Xóa matches cũ hơn 30 ngày và đã kết thúc
-- DELETE FROM matches 
-- WHERE finished_at IS NOT NULL 
--   AND finished_at < NOW() - INTERVAL '30 days';

-- Xóa coin transactions cũ hơn 1 năm
-- DELETE FROM coin_transactions 
-- WHERE created_at < NOW() - INTERVAL '1 year';

-- Update expired subscriptions
-- UPDATE premium_subscriptions 
-- SET status = 'expired', updated_at = NOW()
-- WHERE status = 'active' 
--   AND expires_at < NOW();

-- Vacuum database (chạy khi cần)
-- VACUUM ANALYZE;

\echo ''
\echo '💡 Uncomment các câu lệnh trên để chạy maintenance'
\echo ''

