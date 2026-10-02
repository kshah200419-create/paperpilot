# PaperPilot developer commands. Run `make help` for a list.
SHELL := /bin/bash
BACKEND := backend
FRONTEND := frontend
UV := uv run --project $(BACKEND)
export DATABASE_URL ?= postgresql+asyncpg://postgres:postgres@localhost:5432/paperpilot

.DEFAULT_GOAL := help
.PHONY: help install db migrate seed ingest api web dev test test-backend test-frontend lint \
        typecheck format eval up down logs screenshots clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install backend (with local embeddings) and frontend dependencies
	cd $(BACKEND) && uv sync --extra ml --extra dev
	cd $(FRONTEND) && npm ci
	-$(UV) pre-commit install

db: ## Start only Postgres + pgvector in Docker
	docker compose up -d db

migrate: ## Apply database migrations
	$(UV) paperpilot db upgrade

seed: migrate ## Load the bundled 480-paper sample and build topics
	$(UV) paperpilot seed

ingest: ## Incrementally ingest from arXiv, e.g. `make ingest CAT=cs.IR MAX=100`
	$(UV) paperpilot ingest --category $(or $(CAT),cs.LG) --max $(or $(MAX),100)

api: ## Run the API with autoreload on :8000
	$(UV) uvicorn paperpilot.api.main:app --reload --port 8000

web: ## Run the Vite dev server on :5173 (proxies /api to :8000)
	cd $(FRONTEND) && npm run dev

test: test-backend test-frontend ## Run all tests

test-backend: ## Backend tests (Postgres tests auto-skip if DB is unreachable)
	cd $(BACKEND) && uv run pytest --cov --cov-report=term

test-frontend: ## Frontend tests
	cd $(FRONTEND) && npm test

lint: ## Lint backend (ruff) and frontend (eslint + prettier)
	cd $(BACKEND) && uv run ruff check . && uv run ruff format --check .
	cd $(FRONTEND) && npm run lint && npm run format:check

typecheck: ## mypy --strict and tsc
	cd $(BACKEND) && uv run mypy
	cd $(FRONTEND) && npm run typecheck

format: ## Auto-format everything
	cd $(BACKEND) && uv run ruff check --fix . && uv run ruff format .
	cd $(FRONTEND) && npm run format

eval: ## Retrieval evaluation (keyword vs vector vs hybrid vs rerank)
	$(UV) paperpilot eval --rerank --output eval/results/latest.json

up: ## Build and start the whole stack (http://localhost:8080)
	docker compose up --build -d
	@echo "Web: http://localhost:$${WEB_PORT:-8080}  API docs: http://localhost:$${API_PORT:-8000}/docs"

down: ## Stop the stack
	docker compose down

logs: ## Tail container logs
	docker compose logs -f api web

screenshots: ## Capture UI screenshots (needs running web + api)
	cd $(FRONTEND) && npm run screenshots

clean: ## Remove caches and build outputs
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache $(BACKEND)/htmlcov
	rm -rf $(FRONTEND)/dist $(FRONTEND)/coverage
