PYTHON ?= python3
VENV_PYTHON ?= ./venv/bin/python3
NPM ?= npm

.PHONY: setup setup-backend setup-frontend backend frontend dev clean

setup: setup-backend setup-frontend

setup-backend:
	$(VENV_PYTHON) -m pip install -r backend/requirements.txt

setup-frontend:
	cd frontend && $(NPM) install

backend:
	cd backend && PYTHONPATH="$(PWD)/backend" "$(PWD)/venv/bin/python3" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && $(NPM) run dev -- --host 0.0.0.0 --port 5173

dev:
	./start.sh

clean:
	rm -rf frontend/dist .pytest_cache htmlcov
