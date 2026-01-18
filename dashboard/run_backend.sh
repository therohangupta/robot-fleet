#!/bin/bash
# Run the Dashboard Backend Server

set -e

# Navigate to the project root
cd "$(dirname "$0")/.."

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate multirobot 2>/dev/null || true
fi

# Install dashboard dependencies if needed
pip install -q fastapi uvicorn websockets pydantic

# Run the FastAPI server
echo "🚀 Starting Dashboard Backend on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""

cd dashboard
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
