#!/usr/bin/env bash
set -euo pipefail

DOMAIN_BASE="example.com"
APP_SUBDOMAIN="app"
API_SUBDOMAIN="api"
APP_DIR="/opt/spotify-optimizer"
RUN_USER="www-data"
NODE_MAJOR="20"
REPO_URL="https://github.com/PRADDZY/spotify-optimizer.git"

if [[ "$DOMAIN_BASE" == "example.com" ]]; then
  echo "Edit DOMAIN_BASE, APP_SUBDOMAIN, API_SUBDOMAIN in deploy/setup_vps.sh before running."
  exit 1
fi

if ! id "$RUN_USER" >/dev/null 2>&1; then
  echo "RUN_USER does not exist: $RUN_USER"
  exit 1
fi

APP_DOMAIN="${APP_SUBDOMAIN}.${DOMAIN_BASE}"
API_DOMAIN="${API_SUBDOMAIN}.${DOMAIN_BASE}"

sudo apt update
sudo apt install -y nginx redis-server git python3-venv python3-pip curl certbot python3-certbot-nginx

curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo -E bash -
sudo apt install -y nodejs

sudo mkdir -p "$APP_DIR"
sudo chown "$USER":"$USER" "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ../frontend
npm install
npm run build

sudo mkdir -p "$APP_DIR/backend/data" "$APP_DIR/backend/cache" "$APP_DIR/backend/logs" "$APP_DIR/backend/models"
sudo chown -R "$RUN_USER":"$RUN_USER" \
  "$APP_DIR/backend/data" \
  "$APP_DIR/backend/cache" \
  "$APP_DIR/backend/logs" \
  "$APP_DIR/backend/models"

sudo cp "$APP_DIR/deploy/systemd/spotify-backend.service" /etc/systemd/system/spotify-backend.service
sudo cp "$APP_DIR/deploy/systemd/spotify-frontend.service" /etc/systemd/system/spotify-frontend.service

sudo sed -i "s|/opt/spotify-optimizer|$APP_DIR|g" /etc/systemd/system/spotify-backend.service
sudo sed -i "s|/opt/spotify-optimizer|$APP_DIR|g" /etc/systemd/system/spotify-frontend.service
sudo sed -i "s|www-data|$RUN_USER|g" /etc/systemd/system/spotify-backend.service
sudo sed -i "s|www-data|$RUN_USER|g" /etc/systemd/system/spotify-frontend.service

sudo systemctl daemon-reload
sudo systemctl enable --now spotify-backend spotify-frontend

sudo cp "$APP_DIR/deploy/nginx/spotify-optimizer.conf" /etc/nginx/sites-available/spotify-optimizer
sudo sed -i "s|app.example.com|$APP_DOMAIN|g" /etc/nginx/sites-available/spotify-optimizer
sudo sed -i "s|api.example.com|$API_DOMAIN|g" /etc/nginx/sites-available/spotify-optimizer

sudo ln -sf /etc/nginx/sites-available/spotify-optimizer /etc/nginx/sites-enabled/spotify-optimizer
sudo nginx -t
sudo systemctl reload nginx

sudo certbot --nginx -d "$APP_DOMAIN" -d "$API_DOMAIN"

echo "Done. Remember to configure backend/.env and frontend/.env.local in $APP_DIR."
