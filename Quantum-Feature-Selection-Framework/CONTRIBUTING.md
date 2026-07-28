# Contributing

Thanks for your interest in improving the Quantum Feature Selection Framework.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/short-description
   ```

## Development guidelines

- Keep `src/` free of Streamlit-specific code — dashboard-only logic belongs in `dashboard/` or `src/ui_components.py`.
- Follow the existing docstring style (module summary + `Parameters`/`Returns` sections) for any new public function.
- Add or update tests in `tests/` for any change to `src/`.
- Run the test suite before opening a pull request:
  ```bash
  pytest
  ```
- Keep pull requests focused on a single change; unrelated formatting or refactors should be separate PRs.

## Reporting issues

When filing a bug report, please include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- The relevant traceback, if any

## Code of conduct

Be respectful and constructive. Assume good intent, and keep discussions focused on the technical merits of a change.
