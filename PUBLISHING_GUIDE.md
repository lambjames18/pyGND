# Publishing Guide for PyGND

This guide explains how PyGND is released to PyPI. Releasing is almost entirely
automated: **pushing a `v*` git tag triggers CI to build and publish the
package**. There is no manual version bumping and no local `twine upload` step.

## How releases work

- **Versioning** is fully automatic via [`hatch-vcs`](https://github.com/ofek/hatch-vcs):
  the version is derived from the latest git tag (e.g. tag `v0.2.0` produces
  version `0.2.0`). Nothing in `pyproject.toml` or `src/pygnd/_version.py`
  needs to be edited by hand — `_version.py` is generated at build time.
- **Publishing** happens in the `release` job of
  [`.github/workflows/cicd.yml`](.github/workflows/cicd.yml), which runs when
  a tag matching `v*` is pushed. It builds the package with `uv build` and
  uploads it to PyPI using
  [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish)
  via **PyPI Trusted Publishing** (OIDC) — no API tokens or `.pypirc` files
  are involved.

## Cutting a release

1. Make sure `main` is green (CI passing) and has everything you want in the
   release.
2. Update `CHANGELOG.md` with a new version section.
3. Tag and push:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
4. Watch the `release` job in the [Actions tab](https://github.com/lambjames18/pyGND/actions/workflows/cicd.yml).
   On success, the new version is live on [PyPI](https://pypi.org/project/pygnd/).
5. Optionally, create a GitHub release from the tag with release notes.

Follow [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`
(e.g. `v0.1.0` → `v0.1.1` bug fix → `v0.2.0` new feature → `v1.0.0` stable API).

## One-time setup: PyPI Trusted Publisher

Trusted publishing must be configured once on PyPI before the first tagged
release can succeed (this step must be done by a maintainer with access to
the PyPI project/account — it cannot be done from CI).

1. Go to https://pypi.org/manage/account/publishing/ (for a project that
   doesn't exist on PyPI yet, this is the "pending publisher" flow).
2. Add a new pending publisher with:
   - **PyPI Project Name**: `pygnd`
   - **Owner**: `lambjames18`
   - **Repository name**: `pyGND`
   - **Workflow name**: `cicd.yml`
   - **Environment name**: leave blank (the `release` job does not use a
     GitHub Environment)
3. Save. The next `v*` tag pushed to the repository will be able to publish
   successfully.

## Testing a build locally

You can build and sanity-check the package without publishing anything:

```bash
uv build
pip install dist/pygnd-*.whl
python -c "import pygnd; print(pygnd.__version__)"
cd examples
python example_ang_run.py
```

## Common Issues

### Import errors after installation
- Ensure the package structure under `src/pygnd/` is correct.
- Check that `__init__.py` properly exports the expected functions.

### Missing dependencies
- Verify all dependencies are listed in `pyproject.toml`.

### `release` job fails with an authentication/trusted-publisher error
- Confirm the pending publisher (or, after the first successful publish, the
  regular trusted publisher) is configured on PyPI exactly as above —
  project name, owner, repository, and workflow filename must match exactly.

### Tag pushed but no release happened
- Confirm the tag matches the `v*` pattern (e.g. `v0.1.0`, not `0.1.0`) and
  was pushed to `origin` (`git push origin <tag>`), not just created locally.
