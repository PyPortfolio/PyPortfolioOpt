## Makefile for config-templates: developer tasks orchestrated via go-task
#
# This Makefile wraps the Taskfile.yml commands and provides a friendly
# `make help` index. Lines with `##` after a target are parsed into help text,
# and lines starting with `##@` create section headers in the help output.
#
# Colors for pretty output in help messages
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m

# Default goal when running `make` with no target
.DEFAULT_GOAL := help

# Declare phony targets (they don't produce files)
.PHONY: install-task install clean test marimo book fmt deptry help all

UV_INSTALL_DIR := "./bin"

##@ Bootstrap
install-task: ## ensure go-task (Taskfile) is installed
	@mkdir -p ${UV_INSTALL_DIR}

	@if [ ! -x "${UV_INSTALL_DIR}/task" ]; then \
		printf "$(BLUE)Installing go-task (Taskfile)$(RESET)\n"; \
		curl --location https://taskfile.dev/install.sh | sh -s -- -d -b ${UV_INSTALL_DIR}; \
	fi


install: install-task ## install
	@./bin/task build:install --silent

clean: install-task ## clean
	@./bin/task cleanup:clean --silent

##@ Development and Testing
test: install-task ## run all tests
	@./bin/task docs:test --silent

marimo: install-task ## fire up Marimo server
	@./bin/task docs:marimo --silent

##@ Documentation
book: install-task ## compile the companion book
	@./bin/task docs:test --silent
	@./bin/task docs:docs --silent
	@./bin/task docs:marimushka --silent
	@./bin/task docs:book --silent

fmt: install-task ## check the pre-commit hooks and the linting
	@./bin/task quality:lint --silent

deptry: install-task ## run deptry if pyproject.toml exists
	@if [ -f pyproject.toml ]; then \
		./bin/task build:uv --silent; \
  		echo "→ Running deptry..."; \
		./bin/task quality:deptry --silent; \
	else \
		echo "${GREEN} ⚠ Skipping deptry (no pyproject.toml)"; \
	fi

all: fmt deptry test book ## Run everything
	echo "Run fmt, deptry, test and book"

##@ Meta
help: ## Display this help message
	+@printf "$(BOLD)Usage:$(RESET)\n"
	+@printf "  make $(BLUE)<target>$(RESET)\n\n"
	+@printf "$(BOLD)Targets:$(RESET)\n"
	+@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
