#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment missing. Run: make setup"
  exit 1
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
