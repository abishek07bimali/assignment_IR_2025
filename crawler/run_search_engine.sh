#!/bin/bash

echo "🚀 Starting Coventry University Publications Search Engine"
echo "==========================================="

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "🔧 Starting FastAPI backend server..."
python3 -m uvicorn search_engine_backend:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Backend started with PID: $BACKEND_PID"
echo "Backend running at: http://localhost:8000"
echo ""

sleep 3

cd google-crawl-softwarica

if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

echo ""
echo "🎨 Starting React frontend..."
npm run dev &
FRONTEND_PID=$!

echo "Frontend started with PID: $FRONTEND_PID"
echo ""
echo "==========================================="
echo "✅ Search Engine is ready!"
echo ""
echo "🌐 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo "==========================================="

trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait