-- ============================================================
-- GoGame Database Reset Script
-- ============================================================
-- Script SQL để reset database (xóa tất cả dữ liệu nhưng giữ schema)
-- ⚠️ CẢNH BÁO: Script này sẽ XÓA TẤT CẢ dữ liệu trong các bảng!
-- Chỉ chạy khi muốn reset dữ liệu nhưng giữ lại cấu trúc
--
-- Usage:
--   psql -U postgres -d gogame -f scripts/database_reset.sql
-- ============================================================

\c gogame

-- Tắt foreign key checks tạm thời (PostgreSQL không có, nhưng có thể dùng transaction)
BEGIN;

-- Xóa dữ liệu từ tất cả các bảng (theo thứ tự để tránh foreign key violations)
TRUNCATE TABLE premium_requests CASCADE;
TRUNCATE TABLE coin_transactions CASCADE;
TRUNCATE TABLE refresh_tokens CASCADE;
TRUNCATE TABLE matches CASCADE;
TRUNCATE TABLE users CASCADE;
TRUNCATE TABLE alembic_version CASCADE;

-- Reset sequences nếu có
-- (UUID không dùng sequences, nhưng nếu có thì reset ở đây)

COMMIT;

\echo ''
\echo '============================================================'
\echo '✅ Database đã được reset (dữ liệu đã bị xóa)'
\echo '============================================================'
\echo ''
\echo '📊 Kiểm tra số lượng records:'
SELECT 
    'users' as table_name, COUNT(*) as count FROM users
UNION ALL
SELECT 'matches', COUNT(*) FROM matches
UNION ALL
SELECT 'refresh_tokens', COUNT(*) FROM refresh_tokens
UNION ALL
SELECT 'coin_transactions', COUNT(*) FROM coin_transactions
UNION ALL
SELECT 'premium_requests', COUNT(*) FROM premium_requests;
\echo ''

