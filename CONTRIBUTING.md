# Contributing

## Obtaining the software

```bash
git clone https://github.com/AlxndrSchroeder/actgpr.git
cd actgpr
poetry install
```

See the [README](README.md) for usage and the full [documentation](https://alxndrschroeder.github.io/actgpr/) for the API reference and tutorial.

## Reporting bugs and suggesting enhancements

Use [GitHub Issues](https://github.com/AlxndrSchroeder/actgpr/issues) for bug
reports and feature/enhancement requests. Please include:

- what you did, what you expected, and what happened instead
- the `actgpr` version (`poetry show actgpr` or `meta.json`'s `actgpr_version`
  field of any run you have on disk)
- a minimal reproduction, where possible

Found a **security vulnerability**? Do not open a public issue — see
[SECURITY.md](SECURITY.md) instead.

Issues, comments, and documentation are in English.

## Contributing changes

This project follows [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. Branch off `main` (one focused change per branch).
2. Make your change.
3. Open a pull request against `main`. CI must pass before merging.

### Requirements for acceptable contributions

Enforced automatically by CI on every pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)):

- **Formatting** — [black](https://black.readthedocs.io/): `poetry run black --check src/ tests/`
- **Linting** — [ruff](https://docs.astral.sh/ruff/): `poetry run ruff check src/ tests/`
- **Tests** — [pytest](https://docs.pytest.org/): `poetry run pytest tests/`
  (unit, integration, and regression tiers all must pass; add tests for new
  behaviour)
- **Documentation** — [Sphinx](https://www.sphinx-doc.org/) builds without
  warnings: `poetry run sphinx-build -W docs docs/build/html` (public
  functions/classes need NumPy-style docstrings)

Run all four locally before opening a pull request — they're the exact
commands CI runs.

## Maintenance status

`actgpr` is actively maintained; see the
[commit history](https://github.com/AlxndrSchroeder/actgpr/commits/main) and
[CHANGELOG.md](CHANGELOG.md) for current activity.
