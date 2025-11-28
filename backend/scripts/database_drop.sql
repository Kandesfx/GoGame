-- ============================================================
-- GoGame Database Drop Script
-- ============================================================
-- Script SQL để xóa toàn bộ database và dữ liệu
-- ⚠️ CẢNH BÁO: Script này sẽ XÓA TẤT CẢ dữ liệu!
-- Chỉ chạy khi muốn reset hoàn toàn database
--
-- Usage:
--   psql -U postgres -f scripts/database_drop.sql
-- ============================================================

\c postgres

-- Xóa database (sẽ xóa tất cả dữ liệu!)
DROP DATABASE IF EXISTS gogame;

\echo '✅ Database gogame đã được xóa!'
\echo '💡 Chạy database_schema.sql để tạo lại database mới'

