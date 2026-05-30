#!/usr/bin/env bash
# Khởi động backend (FastAPI) và frontend (Vite) cùng lúc cho demo.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$ROOT/backend/.env" ]; then
  echo "⚠️  Chưa có backend/.env — sao chép từ .env.example và điền OPENAI_API_KEY."
  cp "$ROOT/backend/.env.example" "$ROOT/backend/.env"
fi

echo "▶ Backend: http://localhost:8000"
( cd "$ROOT/backend" && uvicorn app.main:app --reload --port 8000 ) &
BACK_PID=$!

echo "▶ Frontend: http://localhost:5173"
( cd "$ROOT/frontend" && npm run dev ) &
FRONT_PID=$!

trap "kill $BACK_PID $FRONT_PID 2>/dev/null" EXIT
wait
