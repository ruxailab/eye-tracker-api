#!/usr/bin/env bash
set -euo pipefail

# run_local.sh
# Load either .env (for local/dev) or .env.production (for prod) and run the app locally
# Usage: ./run_local.sh -e dev|prod [--port PORT]

ENVIRONMENT="dev"
PORT="3000"

usage() {
  cat <<EOF
Usage: $0 -e dev|prod [--port PORT]

Options:
  -e ENV    Environment: dev (default) or prod
  --port    Port to run the app on (default: 3000)
  -h        Show this help
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -e) ENVIRONMENT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ "$ENVIRONMENT" == "dev" ]]; then
  ENV_FILE=".env"
else
  ENV_FILE=".env.production"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE"
  exit 1
fi

echo "Loading environment from $ENV_FILE"
# export all variables defined in the file
set -a
source "$ENV_FILE"
set +a

export PORT="$PORT"

echo "Starting local server on 0.0.0.0:$PORT"
python wsgi.py
