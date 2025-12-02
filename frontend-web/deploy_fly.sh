#!/bin/bash
# Script để deploy frontend lên Fly.io

echo "🚀 Deploying frontend to Fly.io..."
echo "📁 Current directory: $(pwd)"
echo "📄 Using: fly.toml"

fly deploy

