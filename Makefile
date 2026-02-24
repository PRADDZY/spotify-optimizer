SHELL := /bin/bash

.PHONY: backend-install backend-run frontend-install frontend-build frontend-run dev start stop restart status

backend-install:
	cd backend && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

backend-run:
	cd backend && . .venv/bin/activate && uvicorn backend.app:app --reload --port 8000

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build

frontend-run:
	cd frontend && npm run dev

dev:
	$(MAKE) -j2 backend-run frontend-run

start:
	sudo systemctl start spotify-backend spotify-frontend

stop:
	sudo systemctl stop spotify-backend spotify-frontend

restart:
	sudo systemctl restart spotify-backend spotify-frontend

status:
	sudo systemctl status spotify-backend spotify-frontend
