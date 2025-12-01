#!/bin/bash
# Script để chạy với Python Windows

PYTHON_WIN="/c/Users/HAI/AppData/Local/Programs/Python/Python312/python.exe"

if [ ! -f "$PYTHON_WIN" ]; then
    echo "❌ Python Windows not found at: $PYTHON_WIN"
    echo "Please update the path in this script or install Python Windows"
    exit 1
fi

echo "Using Python Windows: $PYTHON_WIN"
"$PYTHON_WIN" "$@"

