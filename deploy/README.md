# VPS Deploy

This folder contains templates and a helper script for deploying on a single VPS.

## Quick Start

1. Edit `deploy/setup_vps.sh` and set:

- `DOMAIN_BASE`
- `APP_SUBDOMAIN`
- `API_SUBDOMAIN`
- `APP_DIR`
- `RUN_USER`

2. On the VPS, run:

```bash
bash deploy/setup_vps.sh
```

3. Create `backend/.env` with production values (see `backend/.env.example`).
4. Create `frontend/.env.local` with:

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.your-domain.com
```

## What It Does

- Installs system packages and Node.js
- Clones the repo
- Builds the frontend
- Installs backend dependencies
- Installs systemd services
- Configures Nginx
- Obtains TLS certs via certbot

## Templates

- `deploy/systemd/spotify-backend.service`
- `deploy/systemd/spotify-frontend.service`
- `deploy/nginx/spotify-optimizer.conf`
