# Contributing to PyGND

Thank you for your interest in contributing to PyGND! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lambjames18/pyGND.git
   cd pyGND
   ```

2. Install dependencies with [uv](https://docs.astral.sh/uv/) (the toolchain this project develops with), including the `dev` dependency group (pytest, pylint, pdoc, etc.):
   ```bash
   uv sync --all-extras --dev
   ```

   Without `uv`, an editable install plus the dev tools works too:
   ```bash
   pip install -e .
   pip install pylint pylint-exit pytest pytest-cov pdoc
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Use Google-style docstrings (the project's API docs are built with [pdoc](https://pdoc.dev/) using `-d google`)
- Lint with pylint: `uv run pylint src/pygnd`

## Testing

Before submitting a pull request:

1. Ensure all tests pass: `uv run pytest`
2. Add tests for new features or bug fixes
3. Test with different crystal structures (FCC, BCC, HCP) where relevant
4. Verify that the example scripts in `examples/` still work

## Submitting Changes

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Make your changes
4. Commit with clear, descriptive messages
5. Push to your fork
6. Submit a pull request

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Minimal code to reproduce the issue
- Full error traceback if applicable

## Questions

For questions about usage or development, please open an issue on GitHub.
