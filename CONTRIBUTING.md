# Contributing to quant-lab

Thank you for contributing. This document covers the required workflow.

---

## Development Setup

```bash
git clone https://github.com/babicmarko33/quant-lab.git
cd quant-lab
pip install -e ".[ci]"           # All test + ML + dashboard dependencies
pre-commit install                 # Activate pre-commit hooks
cp .env.example .env               # Add your API keys
```

---

## Workflow: Test-Driven Development (Mandatory)

Every change MUST follow the TDD cycle:

```
1. Write a failing test
2. Run → verify RED
3. Write minimal implementation
4. Run → verify GREEN
5. Refactor if needed → stay GREEN
6. ruff check src/ tests/ scripts/ (zero errors)
7. Commit
```

**No implementation without a failing test first.**

---

## Commit Convention

`<type>(<scope>): <description>`

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `test` | Adding/fixing tests |
| `docs` | Documentation only |
| `refactor` | Restructure without new feature |
| `perf` | Performance improvement |
| `ci` | CI/CD pipeline changes |
| `chore` | Dependencies, config updates |

Scopes: `quantcore`, `alpha-engine`, `alpha-ml`, `portfolio`, `execution`, `dashboard`, `derivatives`, `backtest`

---

## Code Quality Requirements

Before opening a PR:

```bash
# All must pass
ruff check src/ tests/ scripts/      # Zero errors
pytest -m "not integration" -q       # All tests green
```

---

## Adding a New Strategy

1. Subclass `alpha_engine.strategies.base.Strategy`
2. Implement `name: str` (property) and `generate_signals(df: pd.DataFrame) -> pd.Series`
3. Signals must be in `{-1, 0, 1}` — no floats, no look-ahead
4. Register in `alpha_engine.strategies.REGISTRY`
5. Write tests: signal bounds, walk-forward compatibility, smoke test via `strategy.run(df)`
6. Add to `scripts/run_backtest.py`

## Adding a New Allocator

See [docs/sops/portfolio-allocation.md](docs/sops/portfolio-allocation.md).

## Adding a New ML Model

See [docs/sops/ml-signal-pipeline.md](docs/sops/ml-signal-pipeline.md).

---

## Pull Request Guidelines

- Branch from `main`: `git checkout -b feat/<scope>-<description>`
- One concern per PR
- Include test count change in description: "190 -> 205 tests"
- CI must be green before merge

---

## Security

- Never commit API keys, secrets, or `.env` files
- `detect-private-key` pre-commit hook is active
- All broker tests MUST mock the network layer (`unittest.mock.patch`)
- Report security vulnerabilities via GitHub Security Advisories (not issues)
