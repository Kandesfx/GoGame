#!/bin/bash
# Script để activate virtual environment và chạy parse script

# Activate venv
source venv/bin/activate

# Run the script with all arguments
python scripts/parse_sgf_local.py "$@"

