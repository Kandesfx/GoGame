#!/bin/bash
# Script để deploy backend lên Fly.io
# Chạy từ root directory của project

cd "$(dirname "$0")/.." || exit 1

echo "🚀 Deploying backend to Fly.io..."
echo "📁 Build context: $(pwd)"
echo "📄 Using: backend/fly.toml"

fly deploy -c backend/fly.toml

