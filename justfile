# check style, typing, and run tests
check-all: check-static test

# check style and typing
check-static: lint typecheck

# check style and format
lint:
    uv run --group=lint -- ruff check --extend-select I .
    uv run --group=lint -- ruff format --check .

# format code and sort imports
format:
    uv run --group=lint -- ruff check --select I --fix .
    uv run --group=lint -- ruff format .

# check static typing annotations
typecheck:
    uv run --group=typing -- mypy stagpy/ tests/

# run test suite
test:
    uv run --group=test -- pytest --cov=./stagpy --cov-report term-missing

# invoke mkdocs with appropriate dependencies
mkdocs *FLAGS:
    uv run --group=doc -- mkdocs {{FLAGS}}

# prepare a new release
release version:
    @if [ -n "$(git status --porcelain || echo "dirty")" ]; then echo "repo is dirty!"; exit 1; fi
    sed -i 's/^version = ".*"$/version = "{{ version }}"/g' pyproject.toml
    git add pyproject.toml
    sed -i 's/"stagpy~=.*"/"stagpy~={{ version }}"/g' docs/user-guide/install.md
    git add docs/user-guide/install.md
    git commit -m "release {{ version }}"
    git tag -m "Release {{ version }}" -a -e "v{{ version }}"
    @echo "check last commit and ammend as necessary, then git push --follow-tags"
