.PHONY: demo install test lint clean

# One-command demo: starts services, ingests docs, runs sample queries
demo:
	@echo "=== Miles RAG Demo ==="
	@echo ""
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "ERROR: ANTHROPIC_API_KEY environment variable is required."; \
		echo ""; \
		echo "Get an API key from https://console.anthropic.com/"; \
		echo "Then run: export ANTHROPIC_API_KEY='your-key-here'"; \
		echo "Or create a .env file with: ANTHROPIC_API_KEY=your-key-here"; \
		exit 1; \
	fi
	@python demo.py

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Run linter
lint:
	ruff check .
	ruff format --check .

# Format code
format:
	ruff check --fix .
	ruff format .

# Clean up
clean:
	docker compose down -v 2>/dev/null || true
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
