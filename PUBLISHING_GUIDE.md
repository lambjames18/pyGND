# Publishing Guide for PyGND

This guide explains how to publish PyGND to PyPI and maintain the package.

## Pre-Publishing Checklist

Before publishing, ensure:

- [ ] Update version number in `src/pygnd/_version.py` and `pyproject.toml`
- [ ] Update README.md with any new features or changes
- [ ] Update your email in `pyproject.toml`
- [ ] Update GitHub repository URL in `pyproject.toml` and `README.md`
- [ ] All tests pass
- [ ] LICENSE file is correct
- [ ] CHANGELOG.md is updated (create if needed)

## Building the Package

1. Install build tools:
   ```bash
   pip install build twine
   ```

2. Build the package:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/pygnd-0.1.0.tar.gz` (source distribution)
   - `dist/pygnd-0.1.0-py3-none-any.whl` (wheel)

3. Check the package:
   ```bash
   twine check dist/*
   ```

## Testing the Package Locally

Before publishing, test the package locally:

```bash
# Install from local build
pip install dist/pygnd-0.1.0-py3-none-any.whl

# Test import
python -c "import pygnd; print(pygnd.__version__)"

# Run example scripts
cd examples
python example_ang_run.py
```

## Publishing to Test PyPI (Recommended First)

1. Create an account on [Test PyPI](https://test.pypi.org/account/register/)

2. Configure `.pypirc` (in your home directory):
   ```ini
   [testpypi]
   username = __token__
   password = pypi-your-test-token-here
   ```

3. Upload to Test PyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```

4. Test installation from Test PyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ pygnd
   ```

## Publishing to PyPI

1. Create an account on [PyPI](https://pypi.org/account/register/)

2. Configure `.pypirc`:
   ```ini
   [pypi]
   username = __token__
   password = pypi-your-production-token-here
   ```

3. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

4. Verify installation:
   ```bash
   pip install pygnd
   ```

## Post-Publishing

1. Tag the release in Git:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. Create a GitHub release with the tag

3. Update documentation if needed

## Versioning

Follow [Semantic Versioning](https://semver.org/):
- MAJOR version: incompatible API changes
- MINOR version: backwards-compatible functionality
- PATCH version: backwards-compatible bug fixes

Example: `0.1.0` → `0.1.1` (bug fix) → `0.2.0` (new feature) → `1.0.0` (stable API)

## Updating the Package

For subsequent releases:

1. Update version in `src/pygnd/_version.py` and `pyproject.toml`
2. Clean old builds: `rm -rf dist/ build/ src/pygnd.egg-info/`
3. Build new version: `python -m build`
4. Upload: `twine upload dist/*`
5. Tag in Git: `git tag -a v0.2.0 -m "Release version 0.2.0"`

## Common Issues

### Import errors after installation
- Ensure package structure is correct
- Check that `__init__.py` properly exports functions

### Missing dependencies
- Verify all dependencies are listed in `pyproject.toml`

### Files not included in package
- Update `MANIFEST.in` to include necessary files
- Check `pyproject.toml` package discovery settings
