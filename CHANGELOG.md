# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
