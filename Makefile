.PHONY: test test-verbose test-coverage lint format install clean

# Run unit tests
test:
	uv run pytest tests/

# Run tests with verbose output
test-verbose:
	uv run pytest tests/ -v

# Run tests with coverage report
test-coverage:
	uv run pytest tests/ --cov=src --cov-report=term-missing

# Run snapshot tests and update snapshots
test-update-snapshots:
	uv run pytest tests/ --snapshot-update

# Lint code
lint:
	uv run ruff check src/ tests/

# Format code
format:
	uv run ruff format src/ tests/

# Install dependencies
install:
	uv sync

# Run the MCP server
serve:
	uv run python -m src.infra.mcp.server

# Clean up cache files
clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/**/__pycache__
	rm -rf tests/__pycache__
	rm -rf .ruff_cache
