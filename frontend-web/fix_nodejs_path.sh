#!/bin/bash
# Script để fix Node.js path trong Git Bash

echo "🔍 Checking Node.js installation..."

# Fix NODE_OPTIONS issue first
unset NODE_OPTIONS

# Check common Node.js locations
NODE_PATHS=(
  "/c/Program Files/nodejs"
  "/c/Program Files (x86)/nodejs"
  "/c/Users/$USER/AppData/Roaming/npm"
  "$HOME/AppData/Roaming/npm"
)

NODE_FOUND=""

for path in "${NODE_PATHS[@]}"; do
  if [ -f "$path/node.exe" ]; then
    NODE_FOUND="$path"
    echo "✅ Found Node.js at: $path"
    break
  fi
done

if [ -z "$NODE_FOUND" ]; then
  echo "❌ Node.js not found in common locations"
  echo ""
  echo "Please check:"
  echo "1. Open Windows Command Prompt (cmd.exe)"
  echo "2. Run: where node"
  echo "3. Add that path to your PATH environment variable"
  exit 1
fi

# Add to PATH for current session
export PATH="$NODE_FOUND:$PATH"

# Verify
if command -v node &> /dev/null; then
  echo "✅ Node.js is now available!"
  echo "   Node version: $(node --version 2>&1)"
  echo "   npm version: $(npm --version 2>&1)"
  echo ""
  echo "⚠️  This is only for current session."
  echo ""
  echo "To make it permanent, add these lines to your ~/.bashrc:"
  echo "   unset NODE_OPTIONS"
  echo "   export PATH=\"$NODE_FOUND:\$PATH\""
  echo ""
  echo "Or run this command:"
  echo "   echo 'unset NODE_OPTIONS' >> ~/.bashrc"
  echo "   echo 'export PATH=\"$NODE_FOUND:\$PATH\"' >> ~/.bashrc"
  echo ""
  echo "Then restart Git Bash or run: source ~/.bashrc"
else
  echo "❌ Still not working. Please check PATH manually."
fi

