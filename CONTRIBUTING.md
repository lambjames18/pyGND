# Contributing to PyGND

Thank you for your interest in contributing to PyGND! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TriBeam_GND.git
   cd TriBeam_GND
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Keep lines under 100 characters
- Use Black for code formatting: `black src/`
- Use Ruff for linting: `ruff check src/`

## Testing

Before submitting a pull request:

1. Ensure your code passes all tests (when available)
2. Test with different crystal structures (FCC, BCC, HCP)
3. Verify that example scripts still work

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
