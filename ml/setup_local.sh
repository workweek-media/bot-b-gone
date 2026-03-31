#!/bin/bash
# Bot-B-Gone ML — Local Setup Script
# Run this on your Mac to set up the local environment for autoresearch iterations.

set -e

echo "=== Bot-B-Gone ML Local Setup ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install --upgrade lightgbm xgboost scikit-learn pandas numpy

# Verify data file
DATA_FILE="data/soft_labeled.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: $DATA_FILE not found."
    echo "Copy it from the sandbox or download from BigQuery."
    exit 1
fi

ROW_COUNT=$(wc -l < "$DATA_FILE" | tr -d ' ')
FILE_SIZE=$(ls -lh "$DATA_FILE" | awk '{print $5}')
echo ""
echo "Data file: $DATA_FILE"
echo "  Rows: $ROW_COUNT"
echo "  Size: $FILE_SIZE"

if [ "$ROW_COUNT" -lt 4000000 ]; then
    echo "WARNING: Expected ~4.37M rows but found $ROW_COUNT."
    echo "The file may still be copying. Wait for it to complete."
fi

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo ""
    echo "Initializing git repository..."
    git init
    git add -A
    git commit -m "Initial commit: Bot-B-Gone ML autoresearch setup"
fi

# Verify the pipeline runs
echo ""
echo "=== Running verification test ==="
python3 train.py 2>&1 | tee run.log

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the autoresearch loop with Cursor/Claude Code:"
echo "  1. Open this directory in Cursor"
echo "  2. Point the agent at program.md"
echo "  3. Tell it: 'Read program.md and run the experiment loop. Start with experiment 46.'"
echo ""
echo "Or run manually:"
echo "  python3 train.py 2>&1 | tee run.log"
echo "  grep 'Composite' run.log"
