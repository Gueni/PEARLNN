# PEARLNN Makefile
# Development and maintenance commands for the PEARLNN project

# Configuration
PYTHON := python3
PIP := pip3
PROJECT_NAME := pearlnn
VENV := venv
SOURCE_DIR := pearlnn
TESTS_DIR := pearlnn/tests
SCRIPTS_DIR := scripts
DOCS_DIR := docs
CONFIG_DIR := config

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Development

.PHONY: install
install: ## Install the package in development mode
	@echo "$(GREEN)Installing PEARLNN in development mode...$(NC)"
	$(PIP) install -e .[dev]

.PHONY: install-deps
install-deps: ## Install only dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -e .[dev]

.PHONY: install-docs
install-docs: ## Install documentation dependencies
	@echo "$(GREEN)Installing documentation dependencies...$(NC)"
	$(PIP) install -e .[docs]

.PHONY: setup
setup: ## Set up complete development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@$(MAKE) install-dev
	@$(MAKE) pre-commit-install
	@$(MAKE) test
	@echo "$(GREEN)Development environment ready!$(NC)"

##@ Testing

.PHONY: test
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR) -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR) --cov=$(SOURCE_DIR) --cov-report=html --cov-report=term

.PHONY: test-fast
test-fast: ## Run tests quickly (no coverage)
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR) --exitfirst -x

.PHONY: test-component
test-component: ## Test specific component (set COMPONENT=name)
	@echo "$(GREEN)Testing component: $(COMPONENT)$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR)/test_components.py -k "$(COMPONENT)" -v

.PHONY: test-core
test-core: ## Test core functionality
	@echo "$(GREEN)Testing core functionality...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR)/test_core.py -v

.PHONY: test-community
test-community: ## Test community features
	@echo "$(GREEN)Testing community features...$(NC)"
	$(PYTHON) -m pytest $(TESTS_DIR)/test_community.py -v

##@ Code Quality

.PHONY: format
format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black $(SOURCE_DIR) $(SCRIPTS_DIR)
	isort $(SOURCE_DIR) $(SCRIPTS_DIR)

.PHONY: lint
lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(NC)"
	@$(MAKE) lint-flake8
	@$(MAKE) lint-black
	@$(MAKE) lint-isort
	@$(MAKE) lint-mypy

.PHONY: lint-flake8
lint-flake8: ## Run flake8
	@echo "$(GREEN)Running flake8...$(NC)"
	flake8 $(SOURCE_DIR) $(SCRIPTS_DIR)

.PHONY: lint-black
lint-black: ## Check code formatting with black
	@echo "$(GREEN)Checking code formatting...$(NC)"
	black --check $(SOURCE_DIR) $(SCRIPTS_DIR)

.PHONY: lint-isort
lint-isort: ## Check import sorting with isort
	@echo "$(GREEN)Checking import sorting...$(NC)"
	isort --check-only $(SOURCE_DIR) $(SCRIPTS_DIR)

.PHONY: lint-mypy
lint-mypy: ## Run type checking with mypy
	@echo "$(GREEN)Running type checking...$(NC)"
	mypy $(SOURCE_DIR)

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	pre-commit install

.PHONY: pre-commit-run
pre-commit-run: ## Run pre-commit on all files
	@echo "$(GREEN)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files

.PHONY: pre-commit-update
pre-commit-update: ## Update pre-commit hooks
	@echo "$(GREEN)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate

##@ Documentation

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	mkdocs serve

.PHONY: docs-build
docs-build: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	mkdocs build

.PHONY: docs-deploy
docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(GREEN)Deploying documentation...$(NC)"
	mkdocs gh-deploy

.PHONY: docs-clean
docs-clean: ## Clean documentation build
	@echo "$(GREEN)Cleaning documentation build...$(NC)"
	rm -rf site/

##@ Building and Distribution

.PHONY: build
build: ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	$(PYTHON) -m build

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/

.PHONY: twine-check
twine-check: ## Check package with twine
	@echo "$(GREEN)Checking package with twine...$(NC)"
	twine check dist/*

.PHONY: upload-test
upload-test: ## Upload to TestPyPI
	@echo "$(GREEN)Uploading to TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*

.PHONY: upload
upload: ## Upload to PyPI
	@echo "$(GREEN)Uploading to PyPI...$(NC)"
	twine upload dist/*

.PHONY: release
release: clean-build build twine-check upload ## Create and upload a release

##@ Model Management

.PHONY: build-models
build-models: ## Build initial models
	@echo "$(GREEN)Building initial models...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/build_model.py

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py

.PHONY: test-cli
test-cli: ## Test CLI functionality
	@echo "$(GREEN)Testing CLI...$(NC)"
	pearlnn --help
	pearlnn extract inductor --help
	pearlnn community --help

##@ Data Management

.PHONY: data-clean
data-clean: ## Clean data directories
	@echo "$(GREEN)Cleaning data directories...$(NC)"
	rm -rf data/cache/* data/temp/*

.PHONY: data-reset
data-reset: ## Reset all data (WARNING: destructive)
	@echo "$(RED)WARNING: This will delete all data$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ $${confirm:-N} = y ] || exit 1
	rm -rf data/models/* data/cache/* data/user_data/* data/temp/*

.PHONY: data-structure
data-structure: ## Create data directory structure
	@echo "$(GREEN)Creating data directory structure...$(NC)"
	mkdir -p data/models data/cache data/user_data data/temp

##@ Utility

.PHONY: clean
clean: clean-build docs-clean data-clean ## Clean all generated files
	@echo "$(GREEN)Cleaning all generated files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

.PHONY: venv
venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created. Activate with: source $(VENV)/bin/activate$(NC)"

.PHONY: update-deps
update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt

.PHONY: check-env
check-env: ## Check development environment
	@echo "$(GREEN)Checking development environment...$(NC)"
	@which python3 || (echo "$(RED)Python 3 not found$(NC)" && exit 1)
	@python3 -c "import sys; print(f'Python {sys.version}')"
	@which pip3 || (echo "$(RED)pip3 not found$(NC)" && exit 1)
	@echo "$(GREEN)Environment check passed!$(NC)"

##@ Help

.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)PEARLNN Development Makefile$(NC)"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  $(YELLOW)make setup$(NC)        Set up complete development environment"
	@echo "  $(YELLOW)make test$(NC)         Run all tests"
	@echo "  $(YELLOW)make lint$(NC)         Run all linters"
	@echo "  $(YELLOW)make format$(NC)       Format code"
	@echo "  $(YELLOW)make build-models$(NC) Build initial models"
	@echo ""

.PHONY: version
version: ## Show version information
	@$(PYTHON) -c "from $(SOURCE_DIR) import __version__; print(f'PEARLNN version: {__version__}')"

.PHONY: status
status: ## Show project status
	@echo "$(BLUE)PEARLNN Project Status$(NC)"
	@echo "Version: $$(make version --silent)"
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Virtual env: $$(if [ -d "$(VENV)" ]; then echo 'Active'; else echo 'Not active'; fi)"
	@echo "Tests: $$(python3 -m pytest --tb=no -q 2>/dev/null && echo '$(GREEN)Passing$(NC)' || echo '$(RED)Failing$(NC)')"
	@echo "Code quality: $$(make lint-flake8 >/dev/null 2>&1 && echo '$(GREEN)Good$(NC)' || echo '$(RED)Issues$(NC)')"

##@ Development Workflows

.PHONY: dev-start
dev-start: ## Start development session
	@echo "$(GREEN)Starting development session...$(NC)"
	@$(MAKE) check-env
	@$(MAKE) test-fast
	@echo "$(GREEN)Development environment ready!$(NC)"

.PHONY: dev-test
dev-test: ## Run development test cycle
	@echo "$(GREEN)Running development test cycle...$(NC)"
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test
	@echo "$(GREEN)All checks passed!$(NC)"

.PHONY: pre-push
pre-push: ## Run checks before pushing
	@echo "$(GREEN)Running pre-push checks...$(NC)"
	@$(MAKE) dev-test
	@$(MAKE) build
	@echo "$(GREEN)Ready to push!$(NC)"

##@ Component Development

.PHONY: new-component
new-component: ## Create a new component template (set NAME=component_name)
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: Please specify component name with NAME=component_name$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Creating new component: $(NAME)$(NC)"
	@cp scripts/templates/component_template.py pearlnn/components/$(NAME).py 2>/dev/null || \
		echo "$(YELLOW)Template not found, creating basic structure$(NC)"
	@echo "from .$(NAME) import $(NAME)Analyzer" >> pearlnn/components/__init__.py
	@echo "$(GREEN)Component $(NAME) created. Don't forget to implement the methods!$(NC)"

# Print fancy banner
.PHONY: banner
banner: ## Display PEARLNN banner
	@echo "$(BLUE)"
	@echo "╔══════════════════════════════════════╗"
	@echo "║               PEARLNN                ║"
	@echo "║   Parameter Extraction And Reverse   ║"
	@echo "║      Learning Neural Network         ║"
	@echo "║         Electronics • AI • Free      ║"
	@echo "╚══════════════════════════════════════╝"
	@echo "$(NC)"