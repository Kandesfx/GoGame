-- SQL script để tạo database và user cho GoGame
-- Chạy script này với quyền superuser (postgres)

-- Tạo database
CREATE DATABASE gogame
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Tạo user (tùy chọn - nếu muốn dùng user riêng)
-- CREATE USER gogame_user WITH PASSWORD 'your_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE gogame TO gogame_user;
-- ALTER USER gogame_user CREATEDB;

-- Kết nối đến database gogame và cấp quyền
\c gogame

-- Tạo schema public nếu chưa có (thường đã có sẵn)
CREATE SCHEMA IF NOT EXISTS public;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;

-- Hiển thị thông tin
\echo '✅ Database gogame đã được tạo thành công!'
\echo '💡 Bạn có thể chạy migrations:'
\echo '   alembic upgrade head'

