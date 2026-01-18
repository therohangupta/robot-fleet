#!/bin/bash
# Run the Dashboard Frontend

set -e

# Navigate to the frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Run the dev server
echo "🚀 Starting Dashboard Frontend on http://localhost:5173"
echo ""
npm run dev
