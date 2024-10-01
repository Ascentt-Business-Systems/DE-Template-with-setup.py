.PHONY: clean

PYTHON_INTERPRETER=python3

SOURCE_DIR=./src
SOURCE_PATH=./src/de_modules
TESTS_DIR=./src/tests
PYTEST_LOG_LEVEL=INFO
PYTEST_COV_MIN=50

DOCKER_CMD=sudo docker
ifeq ($(shell uname),Darwin)  # Mac OS X
DOCKER_CMD=podman
endif

# Load all environment variables from .env
# so that they are preloaded before running any command here
ifneq (,$(wildcard ./.env))
include .env
export
endif

clean:
	find . -type f -name ".DS_Store" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

isort:
	isort .

black:
	black .

flake8:
# Configuration cotrolled by .flake8
	flake8

pylint:
# Configuration cotrolled by .pylintrc
	pylint **/*.py

format: isort black flake8 pylint

# +++++++ +++++++ +++++++
# Dependencies
# +++++++ +++++++ +++++++
setup-local-dev:
	poetry config http-basic.jfrog-tmna ${JFROG_EMAIL} ${JFROG_API_KEY}
	pre-commit install

# WARNING: `make requirements` currently calls `--with test` for `requirements.txt`
# TODO: Remove `--with test` flag once DevOps can call `dev_requirements.txt` for CI
requirements:
	poetry export -f requirements.txt --without-hashes \
		--output requirements.txt
	poetry export -f requirements.txt --without-hashes \
		--output dev_requirements.txt


# +++++++ +++++++ +++++++
# Sphinx documentation
# +++++++ +++++++ +++++++
init-docs:
	poetry install --with docs
	mkdir -p docs
	cd docs && sphinx-quickstart
	touch docs/.nojekyll
	echo "<meta http-equiv=\"refresh\" content=\"0; url=./_build/html/index.html\" />" \
	> docs/index.html

docs: clean
	cd docs && \
	rm -rf source/ && \
	rm -rf app/ && \
	make clean && \
	export PYTHONPATH=../:../${SOURCE_DIR} && \
	sphinx-apidoc -o source ../${SOURCE_PATH} -f && \
	make html

docs-pdf: docs
	cd docs && \
	make latexpdf && \
	cp ./_build/latex/${SPHINX_PROJECT_NAME}.pdf ./_build/html/notes/ 2> /dev/null && \
	cp ./_build/latex/${SPHINX_PROJECT_NAME}.pdf ./notes/ 2> /dev/null

serve-docs:
	cd docs && ${PYTHON_INTERPRETER} -m http.server 8888


# +++++++ +++++++ +++++++
# Sized testing
# +++++++ +++++++ +++++++
define run_tests
	export PYTHONPATH=${SOURCE_DIR} && \
	coverage run --source=${SOURCE_DIR} --omit="*/tests/*,*/common_utils/*" \
		-m pytest -vv --log-level=${PYTEST_LOG_LEVEL} $1 && \
	make -s clean && \
	if [ -n "$2" ]; then \
		coverage report --fail-under=$2 -m; \
	else \
		coverage report -m; \
	fi
endef

test-all:
	clear && $(call run_tests,${TESTS_DIR},${PYTEST_COV_MIN})

test-small:
	$(call run_tests,${TESTS_DIR}/small)

test-medium:
	$(call run_tests,${TESTS_DIR}/medium)

test-large:
	$(call run_tests,${TESTS_DIR}/large)

test-module:
ifndef TEST_PATH
	$(error TEST_PATH is not set. Set TEST_PATH to run it.)
endif
	$(call run_tests,${TEST_PATH})

test-case:
ifndef TEST_PATH
	$(error TEST_PATH is not set. Set TEST_PATH to the module containing the test case.)
endif
ifndef TEST_CASE
	$(error TEST_CASE is not set. Set TEST_CASE to the name of the test case to run.)
endif
	$(call run_tests,$(TEST_PATH) -k "$(TEST_CASE)")