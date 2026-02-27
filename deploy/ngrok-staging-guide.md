# VPS + ngrok Staging Guide (No Domain)

This guide runs backend/frontend on your VPS and exposes both via ngrok for testing from your PC.

## 1. Prerequisites

- VPS with Ubuntu/Debian
- SSH access with sudo
- Spotify app credentials
- ngrok account + auth token

## 2. Install App Dependencies on VPS

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip nodejs npm

cd /opt
sudo git clone https://github.com/PRADDZY/spotify-optimizer.git
sudo chown -R $USER:$USER /opt/spotify-optimizer
cd /opt/spotify-optimizer

python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt

npm --prefix frontend install
npm --prefix frontend run build
```

## 3. Install ngrok on VPS

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install -y ngrok

ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

## 4. Create ngrok Tunnel Config (single process, two tunnels)

```bash
mkdir -p ~/.config/ngrok
cat > ~/.config/ngrok/ngrok.yml <<'YAML'
version: "2"
authtoken: YOUR_NGROK_AUTH_TOKEN
tunnels:
  api:
    proto: http
    addr: 127.0.0.1:8000
  app:
    proto: http
    addr: 127.0.0.1:3000
YAML
```

## 5. Backend + Frontend Env

Create backend env:

```bash
cp /opt/spotify-optimizer/backend/.env.example /opt/spotify-optimizer/backend/.env
```

Set these values in `/opt/spotify-optimizer/backend/.env`:

- `SPOTIFY_CLIENT_ID=...`
- `SPOTIFY_CLIENT_SECRET=...`
- `SPOTIFY_REDIRECT_URI=https://<API_NGROK_URL>/callback`
- `FRONTEND_URLS=https://<APP_NGROK_URL>`
- `FRONTEND_REDIRECT_URL=https://<APP_NGROK_URL>`
- `ENV=production`
- `SESSION_COOKIE_SECURE=true`
- `SESSION_COOKIE_SAMESITE=lax`

Create frontend env:

```bash
cp /opt/spotify-optimizer/frontend/.env.example /opt/spotify-optimizer/frontend/.env.local
```

Set in `/opt/spotify-optimizer/frontend/.env.local`:

- `NEXT_PUBLIC_API_BASE_URL=https://<API_NGROK_URL>`

## 6. Run Backend + Frontend as Services

```bash
sudo cp /opt/spotify-optimizer/deploy/systemd/spotify-backend.service /etc/systemd/system/spotify-backend.service
sudo cp /opt/spotify-optimizer/deploy/systemd/spotify-frontend.service /etc/systemd/system/spotify-frontend.service

sudo systemctl daemon-reload
sudo systemctl enable --now spotify-backend spotify-frontend
```

## 7. Run ngrok (both tunnels in one command)

```bash
ngrok start --all
```

Check tunnel URLs:

```bash
curl -s http://127.0.0.1:4040/api/tunnels
```

Use:

- app URL (for browser): `https://<APP_NGROK_URL>`
- api URL: `https://<API_NGROK_URL>`

## 8. Spotify Dashboard Redirect URI

In Spotify Developer Dashboard, add:

- `https://<API_NGROK_URL>/callback`

It must match exactly.

## 9. Restart Services After URL Changes

If ngrok URL changes, update env files and restart:

```bash
sudo systemctl restart spotify-backend spotify-frontend
```

## 10. Validate

From VPS:

```bash
curl -fsS https://<API_NGROK_URL>/ready
curl -fsS https://<APP_NGROK_URL>/.well-known/health
```

From your PC:

1. Open `https://<APP_NGROK_URL>`
2. Connect Spotify
3. Run optimization

## 11. Optional: Auto-run ngrok via systemd

Create `/etc/systemd/system/ngrok-tunnels.service`:

```ini
[Unit]
Description=ngrok tunnels for spotify optimizer staging
After=network-online.target
Wants=network-online.target

[Service]
User=YOUR_VPS_USER
WorkingDirectory=/home/YOUR_VPS_USER
ExecStart=/usr/bin/ngrok start --all --config /home/YOUR_VPS_USER/.config/ngrok/ngrok.yml
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ngrok-tunnels
sudo systemctl status ngrok-tunnels --no-pager
```
