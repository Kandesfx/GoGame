-- ============================================================
-- GoGame Database Backup Script
-- ============================================================
-- Script SQL để backup database
-- 
-- Usage (từ command line):
--   pg_dump -U postgres -d gogame -f backup_$(date +%Y%m%d_%H%M%S).sql
--   hoặc
--   pg_dump -U postgres -d gogame -F c -f backup_$(date +%Y%m%d_%H%M%S).dump
--
-- Restore:
--   psql -U postgres -d gogame < backup_YYYYMMDD_HHMMSS.sql
--   hoặc
--   pg_restore -U postgres -d gogame backup_YYYYMMDD_HHMMSS.dump
-- ============================================================

\echo '============================================================'
\echo '📦 GoGame Database Backup'
\echo '============================================================'
\echo ''
\echo '💡 Sử dụng pg_dump từ command line:'
\echo ''
\echo 'Backup toàn bộ database:'
\echo '   pg_dump -U postgres -d gogame -f backup.sql'
\echo ''
\echo 'Backup chỉ schema (không có dữ liệu):'
\echo '   pg_dump -U postgres -d gogame --schema-only -f schema.sql'
\echo ''
\echo 'Backup chỉ dữ liệu (không có schema):'
\echo '   pg_dump -U postgres -d gogame --data-only -f data.sql'
\echo ''
\echo 'Backup dạng custom (nén, nhanh hơn):'
\echo '   pg_dump -U postgres -d gogame -F c -f backup.dump'
\echo ''
\echo 'Backup từ xa:'
\echo '   pg_dump -h hostname -U postgres -d gogame -f backup.sql'
\echo ''
\echo '============================================================'
\echo ''

-- Hiển thị thông tin database hiện tại
\echo '📊 Thông tin database:'
SELECT 
    datname as database_name,
    pg_size_pretty(pg_database_size(datname)) as size
FROM pg_database 
WHERE datname = 'gogame';

\echo ''
\echo '📈 Số lượng records trong các bảng:'
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    (SELECT COUNT(*) FROM information_schema.tables t WHERE t.table_schema = schemaname AND t.table_name = tablename) as exists
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

\echo ''

