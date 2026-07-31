# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0]

### Added
- `pygnd` command line entry point: prints the version, an ASCII logo, and a summary of the available console scripts
- Project logo (`resources/pyGND_logo.png`) and a desktop icon for the GUI (`resources/pyGND_icon.ico`)

## [1.0.0]

### Added
- `pygnd_calculate` command line interface, with `dream3d` and `ang` subcommands and a `--dry-run` validation mode
- `calculate_and_save_dream3d` and `calculate_and_save_ang` as explicit, single-format entry points
- Expanded test suite covering `pygnd.core`, `pygnd.io`, `pygnd.rotations`, `pygnd.quaternions`, and the CLI
- API documentation built with pdoc and a test coverage report, both published on every push to `main`
- Automatic, git-tag-driven versioning (`hatch-vcs`) and PyPI publishing via CI trusted publishing

### Fixed
- `src/pygnd/_version.py` is no longer tracked in git; committing it caused the release build to see a "dirty" working tree and publish an incorrect `.devN` version instead of the tagged release version

### Changed
- `calculate_and_save(...)` is now a deprecated wrapper that dispatches to `calculate_and_save_dream3d`/`calculate_and_save_ang` based on the arguments passed
- Various bug fixes found during a full code audit of `pygnd.core`, `pygnd.rotations`, and `pygnd.quaternions`

### Deprecated
- `calculate_and_save(...)` — use `calculate_and_save_dream3d(...)` or `calculate_and_save_ang(...)` instead

## [0.1.0] - 2025-12-17

### Added
- Initial release of PyGND
- Core GND calculation functionality
- Support for FCC, BCC, and HCP crystal structures
- Support for ANG and DREAM.3D file formats
- Parallel processing with configurable CPU cores
- L1 and L2 minimization methods
- Flexible slip system selection
- Progress bar support
- Memory-efficient chunking for large datasets
- Example scripts for both ANG and DREAM.3D workflows
- Comprehensive documentation

### Changed
- Migrated from MATLAB to Python implementation
- Improved performance through parallel processing
- Enhanced memory efficiency with chunking

### Technical Details
- Package structure using modern Python packaging (pyproject.toml)
- Type hints throughout the codebase
- Modular architecture with separate modules for I/O, rotations, and quaternions
