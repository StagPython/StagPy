# check style, typing, and run tests
check-all: check-static test

# check style and typing
check-static: lint typecheck

# check style and format
lint:
    uv run -- ruff check --extend-select I .
    uv run -- ruff format --check .

# format code and sort imports
format:
    uv run -- ruff check --select I --fix .
    uv run -- ruff format .

# check static typing annotations
typecheck:
    uv run -- mypy stagpy/ tests/

# run test suite
test:
    uv run -- pytest --cov=./stagpy --cov-report term-missing

# invoke mkdocs with appropriate dependencies
mkdocs *FLAGS:
    uv run --with-requirements=docs/requirements.txt -- mkdocs {{FLAGS}}
