#!/bin/bash
# Script để chạy frontend với Node.js path fix

# Fix NODE_OPTIONS issue
unset NODE_OPTIONS

# Add Node.js to PATH
export PATH="/c/Program Files/nodejs:$PATH"

# Check if Node.js is available
if ! command -v node &> /dev/null; then
  echo "❌ Node.js not found. Please run: bash fix_nodejs_path.sh"
  exit 1
fi

# Verify Node.js works
if ! node --version &> /dev/null; then
  echo "❌ Node.js found but not working. Check NODE_OPTIONS."
  exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
  echo "📦 Installing dependencies..."
  npm install
fi

# Start dev server
echo "🚀 Starting development server..."
npm run dev

