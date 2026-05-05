# Git Workflow — Standard Operating Procedure

## Branch Strategy

```
main           ← Production-ready, all tests green
  └── develop  ← Integration branch, features merge here first
       ├── feat/strategy-momentum
       ├── feat/ml-xgboost-model
       └── fix/slippage-calculation
```

---

## Commit Convention

Format: `<type>(<scope>): <description>`

### Types
| Type | Usage |
|------|-------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `refactor` | Code change with no new feature or fix |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |
| `data` | Data pipeline or schema changes |
| `research` | Research notebooks or analysis |

### Scopes
`quantcore`, `alpha-engine`, `alpha-ml`, `dashboard`, `backtest`, `data`, `indicators`

### Examples
```
feat(alpha-engine): add momentum strategy with 12-1 lookback
fix(quantcore): correct RSI Wilder's smoothing calculation
test(backtest): add walk-forward validation for momentum strategy
research(alpha-ml): XGBoost feature importance analysis notebook
perf(indicators): vectorize ATR computation with numpy
```

---

## Workflow Rules

1. **Never commit directly to `main`** — Always via PR from develop or feature branch
2. **All commits must pass lint** — `ruff check` + `ruff format --check`
3. **All commits must pass tests** — `pytest -x -m "not slow and not integration"`
4. **Feature branches**: Create from `develop`, merge back to `develop`
5. **Release flow**: `develop` → PR to `main` (requires green CI)

---

## Pre-Commit Checklist

Before every commit:
```bash
ruff check src/ tests/ --fix
ruff format src/ tests/
pytest -x --tb=short -m "not slow and not integration"
```

---

## PR Template

```markdown
## What
[One-sentence description of the change]

## Why
[Motivation — what problem does this solve?]

## Validation
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] No look-ahead bias (if strategy/feature code)
- [ ] Performance metrics documented (if strategy)

## Metrics (if applicable)
- Sharpe (OOS): X.XX
- Max DD: X.X%
- Win Rate: X.X%
```
