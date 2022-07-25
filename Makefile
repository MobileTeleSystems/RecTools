VENV=.venv

ifeq (${OS},Windows_NT)
	BIN=${VENV}/Scripts
else
	BIN=${VENV}/bin
endif


PIP=${BIN}/pip
PYTHON=${BIN}/python

FLAKE=${BIN}/flake8
MYPY=${BIN}/mypy
PYLINT=${BIN}/pylint
ISORT=${BIN}/isort
AUTOPEP8=${BIN}/autopep8
BLACK=${BIN}/black
CODESPELL=${BIN}/codespell
BANDIT=${BIN}/bandit

PYTEST=${BIN}/pytest
COVERAGE=${BIN}/coverage

BENCHMARK=benchmark
SOURCES=rectools
TESTS=tests
REPORTS=reports

# Installation

reports:
	@mkdir ${REPORTS}

.venv:
	@echo "Creating virtualenv...\t\t"
	@virtualenv -q -p python3 ${VENV}

	@echo "Setting pip config..."
	@${PIP} config --site set global.index-url http://rep.msk.mts.ru/artifactory/api/pypi/pypi/simple
	@${PIP} config --site set global.trusted-host rep.msk.mts.ru

	@echo "Upgrading pip...\t"
	@${PIP} install -q pip==21.3

	@echo "Installing requirements...\t"
	@${PIP} install -r requirements-dev.txt

	@echo "Installing benchmark requirements...\t"
	@${PIP} install -r benchmark/requirements.txt

	@echo "[Installed]"

install: .venv


# Linters

.isort:
	@echo "Running isort checks..."
	@${ISORT} --check ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Isort checks finished]"

.black:
	@echo "Running black checks..."
	@${BLACK} --check --diff ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Black checks finished]"

.pylint: reports
	@echo "Running pylint checks..."
	@${PYLINT} ${SOURCES} ${TESTS} ${BENCHMARK}
	@${PYLINT} ${SOURCES} ${TESTS} ${BENCHMARK} > ${REPORTS}/pylint.txt
	@echo "[Pylint checks finished]"

.mypy:
	@echo "Running mypy checks..."
	@${MYPY} ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Mypy checks finished]"

.flake8:
	@echo "Running flake8 checks...\t"
	@${FLAKE} ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Flake8 checks finished]"

.bandit:
	@echo "Running bandit checks...\t"
	@${BANDIT} -q -c bandit.yml -r ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Bandit checks finished]"

.codespell:
	@echo "Running codespell checks...\t"
	@${CODESPELL} ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Codespell checks finished]"


# Fixers & formatters

.isort_fix:
	@echo "Fixing isort..."
	@${ISORT} ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Isort fixed]"

.autopep8_fix:
	@echo "Formatting with autopep8..."
	@${AUTOPEP8} --in-place -r ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Autopep8 fixed]"

.black_fix:
	@echo "Formatting with black..."
	@${BLACK} -q  ${SOURCES} ${TESTS} ${BENCHMARK}
	@echo "[Black fixed]"


# Tests

.pytest:
	@echo "Running pytest checks...\t"
	@PYTHONPATH=. ${PYTEST} ${TESTS}/

.test_metrics:
	@echo "Testing whether models are correct...\t"
	@PYTHONPATH=. ${PYTEST} benchmark/check_metrics/check_model_correctness.py

.doctest:
	@echo "Running doctest checks...\t"
	${PYTEST} --doctest-modules ${SOURCES}

coverage: .venv reports
	@echo "Running coverage..."
	${COVERAGE} run --source ${SOURCES} --module pytest
	${COVERAGE} report
	${COVERAGE} html -d ${REPORTS}/coverage_html
	${COVERAGE} xml -o ${REPORTS}/coverage.xml -i

# Generalization

.autoformat: .isort_fix .autopep8_fix .black_fix
autoformat: .venv .autoformat

.lint: .isort .black .flake8 .codespell .mypy .pylint .bandit
lint: .venv .lint

.test: .pytest .doctest
test: .venv .test

test_metrics: .venv .test_metrics


# Cleaning

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf ${VENV}
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +

reinstall: clean install
