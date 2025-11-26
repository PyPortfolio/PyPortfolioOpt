## Makefile — PyPortfolioOpt developer conveniences
#
# This Makefile exposes common local development tasks and a friendly
# `make help` index.
# Conventions used by the help generator:
#   - Lines with `##` after a target are turned into help text.
#   - Lines starting with `##@` create section headers in the help output.
# This file does not affect the library itself; it only streamlines dev workflows.

# Colors for pretty output in help messages
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m

# Default goal when running `make` with no target
.DEFAULT_GOAL := help

# Declare phony targets (they don't produce files)
.PHONY: install install-uv test fmt

UV_INSTALL_DIR := ./bin

##@ Bootstrap
install-uv: ## ensure uv (and uvx) are installed locally
	@mkdir -p ${UV_INSTALL_DIR}
	@if [ -x "${UV_INSTALL_DIR}/uv" ] && [ -x "${UV_INSTALL_DIR}/uvx" ]; then \
		:; \
	else \
		printf "${BLUE}Installing uv${RESET}\n"; \
		curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=${UV_INSTALL_DIR} sh 2>/dev/null || { printf "${RED}[ERROR] Failed to install uv ${RESET}\n"; exit 1; }; \
	fi

install: install-uv ## install
	@printf "${BLUE}[INFO] Creating virtual environment...${RESET}\n"
	# Create the virtual environment
	@./bin/uv venv --python 3.12 || { printf "${RED}[ERROR] Failed to create virtual environment${RESET}\n"; exit 1; }
	@printf "${BLUE}[INFO] Installing dependencies${RESET}\n"
	@./bin/uv sync --all-extras --frozen || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }


##@ Development and Testing
test: install ## run all tests
	@printf "${BLUE}[INFO] Running tests...${RESET}\n"
	@./bin/uv pip install pytest pytest-cov pytest-html
	@mkdir -p _tests/html-coverage _tests/html-report
	@./bin/uv run pytest tests --cov=pypfopt --cov-report=term --cov-report=html:_tests/html-coverage --html=_tests/html-report/report.html

fmt: install-uv ## check the pre-commit hooks and the linting
	@./bin/uvx pre-commit run --all-files
	@./bin/uvx deptry .

##@ Meta
help: ## Display this help message
	+@printf "$(BOLD)Usage:$(RESET)\n"
	+@printf "  make $(BLUE)<target>$(RESET)\n\n"
	+@printf "$(BOLD)Targets:$(RESET)\n"
	+@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
