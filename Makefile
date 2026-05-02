# This Makefile is for development use only

.PHONY: install dev test benchmark clean lock

install:
	@uv sync --all-extras

dev:
	@uv run trovadb-server

test:
	@uv run pytest tests/

benchmark:
	@uv run pytest benchmarks/ -s

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@rm -rf data/
	@alembic upgrade head

lock:
	@uv lock
