#!/bin/bash
# Script setup database cho GoGame backend (Linux/Mac)

set -e  # Exit on error

echo "============================================================"
echo "🚀 GoGame Database Setup Script (Bash)"
echo "============================================================"
echo ""

# Kiểm tra file .env
if [ ! -f .env ]; then
    echo "⚠️  Không tìm thấy file .env"
    echo "💡 Tạo file .env từ env.example:"
    echo "   cp env.example .env"
    exit 1
fi

echo "✅ Đã tìm thấy file .env"

# Load .env và parse POSTGRES_DSN
source .env 2>/dev/null || true

if [ -z "$POSTGRES_DSN" ]; then
    echo "❌ POSTGRES_DSN không được tìm thấy trong .env"
    exit 1
fi

# Parse DSN (format: postgresql+psycopg://user:password@host:port/database)
DSN_CLEANED=$(echo "$POSTGRES_DSN" | sed 's/postgresql+psycopg:\/\///' | sed 's/postgresql:\/\///')

# Extract components
DB_USER=$(echo "$DSN_CLEANED" | cut -d: -f1)
DB_PASS=$(echo "$DSN_CLEANED" | cut -d: -f2 | cut -d@ -f1)
DB_HOST=$(echo "$DSN_CLEANED" | cut -d@ -f2 | cut -d: -f1)
DB_PORT=$(echo "$DSN_CLEANED" | cut -d: -f3 | cut -d/ -f1)
DB_NAME=$(echo "$DSN_CLEANED" | cut -d/ -f2)

# Default values
DB_USER=${DB_USER:-postgres}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-gogame}

echo ""
echo "📋 Thông tin database:"
echo "   Host: $DB_HOST"
echo "   Port: $DB_PORT"
echo "   Database: $DB_NAME"
echo "   User: $DB_USER"
echo ""

# Kiểm tra psql có sẵn không
if ! command -v psql &> /dev/null; then
    echo "❌ psql không được tìm thấy"
    echo "💡 Cài đặt PostgreSQL client:"
    echo "   Ubuntu/Debian: sudo apt install postgresql-client"
    echo "   macOS: brew install postgresql"
    exit 1
fi

# Kiểm tra kết nối đến PostgreSQL
echo "🔌 Đang kiểm tra kết nối đến PostgreSQL..."
export PGPASSWORD="$DB_PASS"
if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ Không thể kết nối đến PostgreSQL"
    echo "💡 Kiểm tra:"
    echo "   1. PostgreSQL đang chạy"
    echo "   2. Thông tin trong .env đúng"
    echo "   3. User có quyền tạo database"
    exit 1
fi
echo "✅ Đã kết nối đến PostgreSQL"

# Kiểm tra database có tồn tại không
echo ""
echo "💾 Đang kiểm tra database '$DB_NAME'..."
DB_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")

if [ "$DB_EXISTS" = "1" ]; then
    echo "ℹ️  Database '$DB_NAME' đã tồn tại"
else
    echo "📦 Đang tạo database '$DB_NAME'..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"
    echo "✅ Đã tạo database '$DB_NAME'"
fi

# Chạy migrations
echo ""
echo "🔄 Đang chạy migrations..."
if [ -d "venv" ] || [ -d "../venv" ]; then
    # Sử dụng venv nếu có
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source ../venv/bin/activate
    fi
fi

python -m alembic upgrade head

if [ $? -eq 0 ]; then
    echo "✅ Migrations đã chạy thành công"
else
    echo "❌ Lỗi khi chạy migrations"
    echo "💡 Xem backend/migrations/TROUBLESHOOTING.md để biết cách xử lý"
    exit 1
fi

# Kiểm tra kết nối
echo ""
echo "🔍 Đang kiểm tra kết nối database..."
TABLE_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")

if [ $? -eq 0 ]; then
    echo "✅ Kết nối database thành công!"
    echo "📊 Số bảng trong database: $TABLE_COUNT"
    echo ""
    echo "============================================================"
    echo "✅ Database setup hoàn tất!"
    echo "============================================================"
    echo ""
    echo "💡 Bạn có thể chạy backend server:"
    echo "   python -m uvicorn app.main:app --reload"
else
    echo "❌ Có lỗi xảy ra. Vui lòng kiểm tra lại."
    exit 1
fi

