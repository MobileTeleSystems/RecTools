VENV=.venv
REPORTS=.reports

BENCHMARK=benchmark
SOURCES=rectools
TESTS=tests
SCRIPTS=scripts



# Installation

.reports:
	mkdir ${REPORTS}

.venv:
	poetry install -E all --no-root

install: .venv .reports


# Linters

.isort:
	poetry run isort --check ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.black:
	poetry run black --check --diff ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.pylint:
	poetry run pylint --jobs 4 ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.mypy:
	poetry run mypy ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.flake8:
	poetry run flake8 ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.bandit:
	poetry run bandit -q -c bandit.yml -r ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.codespell:
	poetry run codespell ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}


# Fixers & formatters

.isort_fix:
	poetry run isort ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.autopep8_fix:
	poetry run autopep8 --in-place -r ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

.black_fix:
	poetry run black -q  ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}


# Tests

.pytest:
	poetry run pytest ${TESTS} --cov=${SOURCES} --cov-report=xml

.doctest:
	poetry run pytest --doctest-modules ${SOURCES} --ignore=rectools/tools/ann.py

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


# Copyright

copyright:
	poetry run python -m scripts.copyright --check ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}

copyright_fix:
	poetry run python -m scripts.copyright ${SOURCES} ${TESTS} ${SCRIPTS} ${BENCHMARK}


# Cleaning

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf ${VENV}
	rm -rf ${REPORTS}
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '*pytest_cache*' -exec rm -rf {} +

reinstall: clean install
