#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-https://api.example.com}"
APP_URL="${APP_URL:-https://app.example.com}"
BACKEND_SERVICE="${BACKEND_SERVICE:-spotify-backend}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-spotify-frontend}"

echo "Validating systemd services..."
sudo systemctl is-active --quiet "${BACKEND_SERVICE}"
echo "  ${BACKEND_SERVICE}: active"
sudo systemctl is-active --quiet "${FRONTEND_SERVICE}"
echo "  ${FRONTEND_SERVICE}: active"

echo "Validating API readiness..."
curl -fsS "${API_URL}/ready" >/dev/null
echo "  ${API_URL}/ready: ok"

echo "Validating frontend health endpoint..."
curl -fsS "${APP_URL}/.well-known/health" >/dev/null
echo "  ${APP_URL}/.well-known/health: ok"

echo "Validation complete."
