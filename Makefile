SHELL := /bin/bash

PROJECT_DIR := $(CURDIR)
BACKEND_DIR := $(PROJECT_DIR)/backend
FRONTEND_DIR := $(PROJECT_DIR)/frontend

PYTHON ?= python3
VENV_PYTHON ?= $(PROJECT_DIR)/venv/bin/python3
PIP ?= $(VENV_PYTHON) -m pip
NPM ?= npm

BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173

.DEFAULT_GOAL := help

.PHONY: help install setup setup-backend setup-frontend backend frontend dev \
	build build-frontend lint lint-frontend verify clean

help:
	@echo "Available targets:"
	@echo "  make setup           Install backend and frontend dependencies"
	@echo "  make setup-backend   Install Python dependencies into ./venv"
	@echo "  make setup-frontend  Install frontend npm dependencies"
	@echo "  make backend         Run FastAPI backend with reload"
	@echo "  make frontend        Run Vite frontend dev server"
	@echo "  make dev             Start backend and frontend together"
	@echo "  make build           Build the frontend production bundle"
	@echo "  make lint            Run frontend lint"
	@echo "  make verify          Run lightweight local checks"
	@echo "  make clean           Remove common generated files"

install: setup

setup: setup-backend setup-frontend

setup-backend:
	@test -x "$(VENV_PYTHON)" || (echo "Missing $(VENV_PYTHON). Create it first with: python3 -m venv venv" && exit 1)
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt

setup-frontend:
	cd $(FRONTEND_DIR) && $(NPM) install

backend:
	@test -x "$(VENV_PYTHON)" || (echo "Missing $(VENV_PYTHON). Create it first with: python3 -m venv venv" && exit 1)
	cd $(BACKEND_DIR) && \
		PYTHONPATH="$(BACKEND_DIR)" \
		"$(VENV_PYTHON)" -m uvicorn main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload

frontend:
	cd $(FRONTEND_DIR) && $(NPM) run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

dev:
	./start.sh

build: build-frontend

build-frontend:
	cd $(FRONTEND_DIR) && $(NPM) run build

lint: lint-frontend

lint-frontend:
	cd $(FRONTEND_DIR) && $(NPM) run lint

verify:
	bash -n ./start.sh
	cd $(FRONTEND_DIR) && $(NPM) run build
	PYTHONPYCACHEPREFIX=/tmp/pycache "$(VENV_PYTHON)" -m py_compile \
		$(BACKEND_DIR)/main.py \
		$(BACKEND_DIR)/core/__init__.py \
		$(BACKEND_DIR)/models/__init__.py \
		$(BACKEND_DIR)/services/__init__.py \
		$(BACKEND_DIR)/routers/__init__.py

clean:
	rm -rf $(FRONTEND_DIR)/dist .pytest_cache htmlcov /tmp/pycache
