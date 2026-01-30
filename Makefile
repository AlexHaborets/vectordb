# This Makefile is for development use only

.PHONY: install dev test benchmark clean

install:
	uv sync --all-extras
	uv add --dev --editable ./sdk

dev:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest tests/

benchmark:
	uv run pytest benchmarks/ -s 

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf data/
	alembic upgrade head

lock:
	uv pip compile pyproject.toml -o requirements.txt