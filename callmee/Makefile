# Makefile for the function calling system

.PHONY: install run debug clean lint

# Install dependencies
install:
	uv sync

# Run the program
run:
	uv run python -m src

# Debug mode
debug:
	uv run python -m pdb -m src

# Clean temporary files
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf data/output/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Lint
lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Run tests
test:
	uv run pytest tests/ -v