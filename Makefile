VENV=.venv

ifeq (${OS},Windows_NT)
	BIN=${VENV}/Scripts
else
	BIN=${VENV}/bin
endif

export PATH := $(BIN):$(PATH)

FLAKE=flake8
MYPY=mypy
PYLINT=pylint
ISORT=isort
AUTOPEP8=autopep8
BLACK=black
CODESPELL=codespell
BANDIT=bandit
PYTEST=pytest
COVERAGE=coverage

BENCHMARK=benchmark
SOURCES=rectools
TESTS=tests
REPORTS=reports

# Installation

reports:
	@mkdir ${REPORTS}

.venv:
	@echo "Creating virtualenv...\t\t"
	poetry install -E all --no-root
	@echo "[Installed]"

install: .venv reports


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
	@PYTHONPATH=. ${PYTEST} ${TESTS} --cov=${SOURCES} --cov-report=xml

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
	@rm -rf ${REPORTS}
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +

reinstall: clean install
