#!/bin/bash
# Setup script for Linux/Mac
# Cài đặt dependencies cho local processing

echo "Installing dependencies for local processing..."
echo

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found! Please install Python 3.8+"
    exit 1
fi

python3 --version

echo
echo "Installing packages..."
python3 -m pip install --upgrade pip
python3 -m pip install sgf numpy torch tqdm

echo
echo "Verifying installation..."
python3 -c "import sgf; import numpy; import torch; import tqdm; print('All packages installed successfully!')"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Some packages failed to install"
    exit 1
fi

echo
echo "Setup complete!"

