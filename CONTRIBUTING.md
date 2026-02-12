# Contributing to Brad

## First read

Read `AGENTS.md` before making changes. It captures mission, constraints, architecture, and active next steps.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,vad]"
```

Optional extras:

```bash
pip install -e ".[llm]"
pip install -e ".[onnx]"
```

## Development workflow

1. Run checks before coding:
   - `brad doctor`
   - `pytest`
   - `ruff check .`
   - `black --check .`
2. Keep changes small and focused.
3. Add or update unit tests.
4. Update docs if behavior changes.
5. Update `CHANGELOG.md`.

## Running quality gates

```bash
ruff check .
black --check .
pytest -q
```

## Pull request guidelines

1. Include a concise problem statement.
2. Explain design choices and tradeoffs.
3. Reference any related learning-path steps.
4. Provide command outputs for lint/tests.
5. Keep runtime local-only. Do not introduce runtime network calls.
