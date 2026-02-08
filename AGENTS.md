# Repository Guidelines

## AGENT のルール
- コード内のコメントは最小限にし, それは英語で記載して下さい. それ以外は日本語で回答して下さい.
- このプロジェクトの構成が変更され次第, プロジェクトに合わせてこの `AGENT.md` も修正して下さい.

## Project Structure & Module Organization
This repository is currently a minimal Python project scaffold.
- `pyproject.toml`: project metadata (`rl-template`), Python requirement (`>=3.13`), and dependency declarations.
- `uv.lock`: lockfile managed by `uv` for reproducible environments.
- `LICENSE`: licensing terms.

As code is added, keep runtime modules in a dedicated package directory (for example `src/rl_template/`) and place tests in `tests/` with a matching module layout.

## Build, Test, and Development Commands
Use `uv` for environment and command execution.
- `uv sync`: create/update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run python -m compileall .`: quick syntax validation across the repository.
- `uv run pytest`: run test suite (after tests are added).

If you add tooling (linting, formatting, type checking), define commands in `pyproject.toml` so contributors can run them consistently through `uv run`.

## Coding Style & Naming Conventions
Target Python 3.13+ and follow PEP 8.
- Indentation: 4 spaces, no tabs.
- Functions/variables/modules: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.

Prefer small, focused modules and explicit imports. Keep public APIs stable and document non-obvious behavior with concise docstrings.

## Testing Guidelines
No test framework is configured beyond Python project defaults yet; use `pytest` when adding tests.
- Put tests in `tests/`.
- Name files `test_*.py` and test functions `test_*`.
- Add regression tests for each bug fix and feature-level tests for new behavior.

## Commit & Pull Request Guidelines
Current history starts with a single `Initial commit`; use clear, imperative commit messages going forward.
- Commit format: short subject line (for example, `Add environment sync docs`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, key changes, test evidence (`uv run pytest` output), and linked issues when applicable.

For user-facing behavior changes, include before/after examples in the PR description.
