#!/bin/bash
# Wrapper script for parse_by_year.py using Windows Python
# This ensures we use the correct Python with all dependencies installed

PYTHON_PATH="/c/Users/Hai/AppData/Local/Programs/Python/Python312/python.exe"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python not found at $PYTHON_PATH"
    echo "Please update PYTHON_PATH in this script to point to your Python installation"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the script with Windows Python
"$PYTHON_PATH" "$SCRIPT_DIR/parse_by_year.py" "$@"

