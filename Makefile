.PHONY: install run test lint format docker-up docker-down benchmark load-test clean

install:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .
	mypy app/

format:
	ruff format .
	ruff check --fix .

docker-up:
	docker compose -f docker-compose.yml up -d --build

docker-down:
	docker compose -f docker-compose.yml down -v

benchmark:
	python scripts/benchmark.py

load-test:
	python scripts/load_test.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
