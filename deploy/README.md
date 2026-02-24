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

## Metrics Auth

The Nginx template protects `/metrics` with basic auth. Create the credentials with:

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd metrics
```

## Nginx Limits

The template sets `client_max_body_size 2m` and conservative proxy buffer sizes for both API and frontend.

## Uptime Check

The frontend serves a static health file at `/.well-known/health`.

## Redis Maintenance

Optional daily Redis health/cleanup task:

```bash
sudo cp deploy/systemd/redis-maintenance.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now redis-maintenance.timer
```

## Redis Backup/Restore

Create backups:

```bash
sudo bash deploy/monitoring/redis_backup.sh
```

Restore from a backup file:

```bash
sudo bash deploy/monitoring/redis_restore.sh /path/to/backup.rdb
```

## Hardening (Fail2ban + UFW)

```bash
sudo apt install -y fail2ban
sudo systemctl enable --now fail2ban

sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## Templates

- `deploy/systemd/spotify-backend.service`
- `deploy/systemd/spotify-frontend.service`
- `deploy/nginx/spotify-optimizer.conf`
- `deploy/monitoring/prometheus-scrape.yml`
- `deploy/monitoring/grafana-dashboard.json`
- `deploy/monitoring/prometheus.yml`
- `deploy/systemd/prometheus.service`
- `deploy/systemd/redis-maintenance.service`
- `deploy/systemd/redis-maintenance.timer`
- `deploy/monitoring/redis_maintenance.sh`
- `deploy/monitoring/redis_backup.sh`
- `deploy/monitoring/redis_restore.sh`
- `deploy/monitoring/grafana-provisioning/datasources/prometheus.yml`
- `deploy/monitoring/grafana-provisioning/dashboards/dashboards.yml`
