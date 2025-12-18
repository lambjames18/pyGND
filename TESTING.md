## Testing Guide for PyGND

This document provides comprehensive information about testing in the PyGND project.

## Overview

The test suite is built using **pytest** and covers the core mathematical operations, data transformations, and validation logic in PyGND. Tests are designed to ensure correctness, numerical stability, and proper error handling.

## Quick Start

```bash
# Install package with test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=pygnd --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared fixtures
├── README.md                # Detailed testing documentation
├── test_core.py             # Core GND calculations (116 lines)
├── test_quaternions.py      # Quaternion operations (233 lines)
├── test_rotations.py        # Rotation conversions (167 lines)
└── test_utils.py            # Utility functions
```

## Test Coverage

### Module: rotations.py

**Functions Tested:**
- `qu2om()` - Quaternion to orientation matrix
- `qu2eu()` - Quaternion to Euler angles
- `qu2ax()` - Quaternion to axis-angle

**Test Coverage:**
- ✅ Identity transformations
- ✅ Known angle rotations (90°, 180°)
- ✅ Batch processing
- ✅ Orthogonality properties
- ✅ Normalization
- ✅ Valid angle ranges
- ✅ Consistency across conversions

**Number of Tests:** 17

### Module: quaternions.py

**Functions Tested:**
- `qu_std()` - Quaternion standardization
- `qu_norm()` - Quaternion normalization
- `qu_prod()` / `qu_prod_raw()` - Quaternion multiplication
- `qu_prod_pos_real()` - Real part extraction
- `qu_triple_prod_pos_real()` - Triple product

**Test Coverage:**
- ✅ Standardization (positive real part)
- ✅ Normalization to unit length
- ✅ Multiplication properties (associativity, identity)
- ✅ Conjugate operations
- ✅ Batch operations
- ✅ Edge cases (zero quaternions, orthogonal quaternions)
- ✅ Idempotency
- ✅ Norm preservation

**Number of Tests:** 31

### Module: core.py

**Functions Tested:**
- `get_linear_operator()` - A and B matrix generation

**Test Coverage:**
- ✅ FCC crystal structure (18 slip systems)
- ✅ BCC crystal structure (all combinations):
  - All (52 systems)
  - screw+110 (16 systems)
  - screw+112 (16 systems)
  - screw+123 (28 systems)
  - screw+110+112 (28 systems)
  - screw+110+123 (40 systems)
  - screw+112+123 (40 systems)
- ✅ HCP crystal structure (all combinations):
  - All (33 systems)
  - Basal (6 systems)
  - Prismatic (3 systems)
  - Pyramidal (24 systems)
  - Combinations
- ✅ Input validation (type and value checking)
- ✅ Case insensitivity
- ✅ Whitespace handling
- ✅ Pseudo-inverse properties (B @ A ≈ I)
- ✅ Matrix properties (rank, condition number)
- ✅ Precision (float32)

**Number of Tests:** 22

### Module: utils.py

**Functions Tested:**
- `tqdm_joblib()` - Context manager for progress bars

**Test Coverage:**
- ✅ Context manager protocol
- ✅ Proper cleanup

**Number of Tests:** 2

## Total Test Count: **72 tests**

## Running Tests

### Basic Commands

```bash
# All tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf
```

### Specific Tests

```bash
# Single file
pytest tests/test_rotations.py

# Single class
pytest tests/test_rotations.py::TestQuaternionToOrientationMatrix

# Single test
pytest tests/test_rotations.py::TestQuaternionToOrientationMatrix::test_identity_quaternion

# Tests matching pattern
pytest -k "identity"
```

### Coverage

```bash
# Generate coverage report
pytest --cov=pygnd

# HTML report
pytest --cov=pygnd --cov-report=html

# Terminal report with missing lines
pytest --cov=pygnd --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=pygnd --cov-fail-under=80
```

### Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration

# List all markers
pytest --markers
```

## Continuous Integration

Tests run automatically on:
- **Push** to main or develop branches
- **Pull requests** to main or develop branches

### CI Matrix

- **Operating Systems:** Ubuntu, macOS, Windows
- **Python Versions:** 3.8, 3.9, 3.10, 3.11, 3.12

### CI Commands

GitHub Actions workflow (`.github/workflows/tests.yml`) runs:
```bash
pytest -v --cov=pygnd --cov-report=xml
```

Coverage is uploaded to Codecov for Python 3.11 on Ubuntu.

## Adding New Tests

### Test Template

```python
"""Tests for new_module."""

import numpy as np
import pytest
from pygnd import new_module


class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        input_data = np.array([1.0, 2.0, 3.0])

        # Act
        result = new_module.new_function(input_data)

        # Assert
        expected = np.array([...])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_invalid_input(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="error message"):
            new_module.new_function(invalid_input)

    def test_edge_case(self):
        """Test edge case."""
        result = new_module.new_function(edge_case_input)
        assert result is not None
```

### Best Practices

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Use descriptive test names (test_what_when_then)
3. **AAA Pattern**: Arrange, Act, Assert
4. **Use Fixtures**: For common setup (see conftest.py)
5. **Numerical Testing**: Use `np.testing.assert_allclose()` with appropriate tolerance
6. **Document**: Add docstrings explaining what is tested
7. **Edge Cases**: Test boundary conditions, empty inputs, etc.
8. **Error Cases**: Test that errors are raised appropriately

## Fixtures

Available fixtures (defined in `conftest.py`):

- `sample_quaternion` - Normalized quaternion [0.5, 0.5, 0.5, 0.5]
- `identity_quaternion` - Identity [1.0, 0.0, 0.0, 0.0]
- `sample_euler_angles` - Sample angles [0.1, 0.2, 0.3]
- `random_seed` - Reproducible random numbers (seed=42)

## Known Issues and Workarounds

### NumPy 2.x Compatibility

Some SciPy versions may have issues with NumPy 2.x:

```bash
# Solution 1: Downgrade NumPy
pip install "numpy<2"

# Solution 2: Upgrade SciPy
pip install --upgrade scipy
```

### Missing PIL (Pillow)

If I/O tests fail due to missing PIL:

```bash
pip install Pillow
```

## Debugging Failed Tests

```bash
# Drop into debugger on failure
pytest --pdb

# Debug specific test
pytest --pdb tests/test_core.py::TestGetLinearOperator::test_fcc_structure

# Show print statements
pytest -s

# Show full diff
pytest -vv
```

## Performance Testing

While not currently implemented, you can add performance markers:

```python
@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset (slow)."""
    # ...
```

Run only fast tests:
```bash
pytest -m "not slow"
```

## Future Improvements

- [ ] Add integration tests with real data files
- [ ] Add property-based testing with Hypothesis
- [ ] Add performance benchmarks
- [ ] Increase coverage to >90%
- [ ] Add mutation testing
- [ ] Add tests for I/O operations
- [ ] Add tests for end-to-end workflows

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [NumPy Testing](https://numpy.org/doc/stable/reference/routines.testing.html)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
