.PHONY: sync sync-cpu format check type test test-fast build
sync:
	uv sync --locked --extra cu128 --group dev
sync-cpu:
	uv sync --locked --extra cpu --group dev
format:
	uv run --no-sync ruff format src tests
	uv run --no-sync ruff check --fix src tests
type:
	uv run --no-sync ty check
	uv run --no-sync pyright
check:
	uv run --no-sync ruff format --check src tests
	uv run --no-sync ruff check src tests
	$(MAKE) type
test:
	MUJOCO_GL=disable uv run --no-sync pytest
test-fast:
	MUJOCO_GL=disable uv run --no-sync pytest -m 'not slow'
build:
	uv build
