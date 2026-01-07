#!/bin/bash
# Unified start script - builds frontend and starts backend server

echo "=========================================="
echo "Starting Deminsify Application"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found. Please run this from the project root."
    exit 1
fi

# Build React frontend
echo "📦 Building React frontend..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

echo "✅ Frontend built successfully!"
echo ""

# Start backend server
echo "🚀 Starting backend server..."
echo "   Server will be available at http://localhost:8000"
echo "   Press CTRL+C to stop"
echo ""

cd backend
if [ -d "venv" ]; then
    source venv/bin/activate
fi

uvicorn main:app --host 0.0.0.0 --port 8000

