#!/bin/bash
# Script để chuẩn bị và push code lên Git

set -e

echo "🔍 Kiểm tra Git repository..."

# Kiểm tra xem đã có git repo chưa
if [ ! -d ".git" ]; then
    echo "⚠️  Chưa có Git repository. Đang khởi tạo..."
    git init
    echo "✅ Đã khởi tạo Git repository"
fi

# Kiểm tra .env files
echo ""
echo "🔍 Kiểm tra file .env..."
ENV_FILES=$(find . -name ".env" -type f 2>/dev/null | grep -v node_modules)
if [ -n "$ENV_FILES" ]; then
    echo "⚠️  Tìm thấy các file .env:"
    echo "$ENV_FILES"
    echo ""
    echo "Kiểm tra xem chúng có được ignore không..."
    for env_file in $ENV_FILES; do
        if git check-ignore -q "$env_file"; then
            echo "✅ $env_file đã được ignore"
        else
            echo "❌ $env_file CHƯA được ignore! Cần thêm vào .gitignore"
            exit 1
        fi
    done
else
    echo "✅ Không tìm thấy file .env"
fi

# Kiểm tra các thư mục lớn
echo ""
echo "🔍 Kiểm tra các thư mục lớn..."
if [ -d "build" ] && ! git check-ignore -q "build/"; then
    echo "⚠️  Thư mục build/ chưa được ignore"
fi
if [ -d "venv" ] && ! git check-ignore -q "venv/"; then
    echo "⚠️  Thư mục venv/ chưa được ignore"
fi
if [ -d "frontend-web/node_modules" ] && ! git check-ignore -q "frontend-web/node_modules/"; then
    echo "⚠️  Thư mục node_modules/ chưa được ignore"
fi

# Kiểm tra git status
echo ""
echo "📊 Git status:"
git status --short | head -20

echo ""
echo "✅ Kiểm tra hoàn tất!"
echo ""
echo "📝 Các bước tiếp theo:"
echo "1. git add ."
echo "2. git commit -m 'Your commit message'"
echo "3. git remote add origin <your-repo-url>  (nếu chưa có)"
echo "4. git push -u origin master  (hoặc main)"

