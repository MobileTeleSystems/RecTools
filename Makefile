VENV=.venv
REPORTS=.reports

BENCHMARK=benchmark
SOURCES=rectools
TESTS=tests



# Installation

.reports:
	mkdir ${REPORTS}

.venv:
	poetry install -E all --no-root

install: .venv .reports


# Linters

.isort:
	poetry run isort --check ${SOURCES} ${TESTS} ${BENCHMARK}

.black:
	poetry run black --check --diff ${SOURCES} ${TESTS} ${BENCHMARK}

.pylint:
	poetry run pylint --jobs 4 ${SOURCES} ${TESTS} ${BENCHMARK}

.mypy:
	poetry run mypy ${SOURCES} ${TESTS} ${BENCHMARK}

.flake8:
	poetry run flake8 ${SOURCES} ${TESTS} ${BENCHMARK}

.bandit:
	poetry run bandit -q -c bandit.yml -r ${SOURCES} ${TESTS} ${BENCHMARK}

.codespell:
	poetry run codespell ${SOURCES} ${TESTS} ${BENCHMARK}


# Fixers & formatters

.isort_fix:
	poetry run isort ${SOURCES} ${TESTS} ${BENCHMARK}

.autopep8_fix:
	poetry run autopep8 --in-place -r ${SOURCES} ${TESTS} ${BENCHMARK}

.black_fix:
	poetry run black -q  ${SOURCES} ${TESTS} ${BENCHMARK}


# Tests

.pytest:
	poetry run pytest ${TESTS} --cov=${SOURCES} --cov-report=xml

.doctest:
	poetry run pytest --doctest-modules ${SOURCES} --ignore=rectools/tools/ann.py --ignore=rectools/models/lightfm.py

coverage: .venv .reports
	poetry run coverage run --source ${SOURCES} --module pytest
	poetry run coverage report
	poetry run coverage html -d ${REPORTS}/coverage_html
	poetry run coverage xml -o ${REPORTS}/coverage.xml -i


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
