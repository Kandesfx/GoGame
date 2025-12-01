#!/bin/bash
# Quick fix và chạy frontend

echo "🔧 Fixing Node.js PATH..."

# Fix NODE_OPTIONS issue
unset NODE_OPTIONS

# Add Node.js to PATH
export PATH="/c/Program Files/nodejs:$PATH"

# Verify
if command -v node &> /dev/null && node --version &> /dev/null; then
  echo "✅ Node.js is working!"
  echo "   Node: $(node --version)"
  echo "   npm: $(npm --version)"
  echo ""
  
  # Check if in frontend-web directory
  if [ ! -f "package.json" ]; then
    echo "⚠️  Not in frontend-web directory. Changing..."
    cd frontend-web 2>/dev/null || {
      echo "❌ frontend-web directory not found"
      exit 1
    }
  fi
  
  # Install dependencies if needed
  if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
  fi
  
  # Start dev server
  echo "🚀 Starting development server..."
  npm run dev
else
  echo "❌ Node.js still not working"
  echo ""
  echo "Try manually:"
  echo "  unset NODE_OPTIONS"
  echo "  export PATH=\"/c/Program Files/nodejs:\$PATH\""
  echo "  node --version"
  exit 1
fi

