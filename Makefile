.PHONY: sync sync-cpu format check stubs type test test-fast build
sync:
	uv sync --locked --extra cu128 --group dev
sync-cpu:
	uv sync --locked --extra cpu --group dev
format:
	uv run --no-sync ruff format src tests scripts
	uv run --no-sync ruff check --fix src tests scripts
stubs:
	bash scripts/generate_mujoco_stubs.sh
type: stubs
	uv run --no-sync ty check
	uv run --no-sync pyright
check:
	uv run --no-sync ruff format --check src tests scripts
	uv run --no-sync ruff check src tests scripts
	$(MAKE) type
test:
	MUJOCO_GL=disable uv run --no-sync pytest
test-fast:
	MUJOCO_GL=disable uv run --no-sync pytest -m 'not slow'
build:
	uv build
