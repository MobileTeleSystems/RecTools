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
REPORTS=.reports


# Installation

.reports:
	mkdir ${REPORTS}

.venv:
	poetry install -E all --no-root

install: .venv .reports


# Linters

.isort:
	${ISORT} --check ${SOURCES} ${TESTS} ${BENCHMARK}

.black:
	${BLACK} --check --diff ${SOURCES} ${TESTS} ${BENCHMARK}

.pylint:
	${PYLINT} --jobs 4 ${SOURCES} ${TESTS} ${BENCHMARK}

.mypy:
	${MYPY} ${SOURCES} ${TESTS} ${BENCHMARK}

.flake8:
	${FLAKE} ${SOURCES} ${TESTS} ${BENCHMARK}

.bandit:
	${BANDIT} -q -c bandit.yml -r ${SOURCES} ${TESTS} ${BENCHMARK}

.codespell:
	${CODESPELL} ${SOURCES} ${TESTS} ${BENCHMARK}


# Fixers & formatters

.isort_fix:
	${ISORT} ${SOURCES} ${TESTS} ${BENCHMARK}

.autopep8_fix:
	${AUTOPEP8} --in-place -r ${SOURCES} ${TESTS} ${BENCHMARK}

.black_fix:
	${BLACK} -q  ${SOURCES} ${TESTS} ${BENCHMARK}


# Tests

.pytest:
	PYTHONPATH=. ${PYTEST} ${TESTS} --cov=${SOURCES} --cov-report=xml

.doctest:
	${PYTEST} --doctest-modules ${SOURCES}

coverage: .venv .reports
	${COVERAGE} run --source ${SOURCES} --module pytest
	${COVERAGE} report
	${COVERAGE} html -d ${REPORTS}/coverage_html
	${COVERAGE} xml -o ${REPORTS}/coverage.xml -i


# Generalization

.format: .isort_fix .autopep8_fix .black_fix
format: .venv .format

.lint: .isort .black .flake8 .codespell .mypy .pylint .bandit
lint: .venv .lint

.test: .pytest .doctest
test: .venv .test


# Cleaning

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf ${VENV}
	rm -rf ${REPORTS}
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '*pytest_cache*' -exec rm -rf {} +

reinstall: clean install
