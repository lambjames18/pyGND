# PyGND Test Suite

This directory contains the test suite for the PyGND package.

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Shared pytest fixtures
├── test_rotations.py     # Tests for rotation conversions
├── test_quaternions.py   # Tests for quaternion operations
├── test_core.py          # Tests for core GND calculations
└── test_utils.py         # Tests for utility functions
```

## Running Tests

### Prerequisites

Install the package with test dependencies:

```bash
# Install in development mode with test dependencies
pip install -e ".[test]"

# Or install all dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=pygnd --cov-report=html
```

### Run Specific Test Files

```bash
# Run tests for a specific module
pytest tests/test_rotations.py

# Run tests for a specific class
pytest tests/test_rotations.py::TestQuaternionToOrientationMatrix

# Run a specific test
pytest tests/test_rotations.py::TestQuaternionToOrientationMatrix::test_identity_quaternion
```

### Test Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

## Test Coverage

The test suite covers:

### Rotations Module (`test_rotations.py`)
- Quaternion to orientation matrix conversion (qu2om)
- Quaternion to Euler angles conversion (qu2eu)
- Quaternion to axis-angle conversion (qu2ax)
- Orthogonality and mathematical properties
- Batch processing

### Quaternions Module (`test_quaternions.py`)
- Quaternion standardization (qu_std)
- Quaternion normalization (qu_norm)
- Quaternion multiplication (qu_prod, qu_prod_raw)
- Real part extraction (qu_prod_pos_real)
- Triple products (qu_triple_prod_pos_real)
- Mathematical properties and invariants

### Core Module (`test_core.py`)
- Linear operator generation (get_linear_operator)
- FCC, BCC, and HCP crystal structures
- All slip system combinations
- Matrix properties (rank, condition number)
- Pseudo-inverse relationships
- Input validation

### Utils Module (`test_utils.py`)
- Context manager protocol
- Integration with joblib and tqdm

## Writing New Tests

### Test Organization

Follow the existing pattern:

```python
class TestFeatureName:
    """Tests for specific feature."""

    def test_basic_case(self):
        """Test description."""
        # Arrange
        input_data = ...

        # Act
        result = function(input_data)

        # Assert
        assert result == expected
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_with_quaternion(sample_quaternion):
    """Example using fixture."""
    result = some_function(sample_quaternion)
    assert result.shape == (4,)
```

### Testing Numerical Code

For numerical computations, use numpy.testing:

```python
import numpy as np

def test_numerical_result():
    result = compute_something()
    expected = np.array([...])
    np.testing.assert_allclose(result, expected, atol=1e-10)
```

### Testing Error Cases

```python
import pytest

def test_invalid_input():
    with pytest.raises(ValueError, match="expected error message"):
        function_with_validation(invalid_input)
```

## Coverage Goals

Target coverage areas:
- Core mathematical functions: >95%
- I/O operations: >80%
- Utility functions: >90%
- Overall: >85%

## Continuous Integration

Tests should pass in all supported Python versions (3.8+) with:
- NumPy 1.20+ and 2.x
- SciPy 1.7+
- All other dependencies as specified in pyproject.toml

## Known Issues

1. **NumPy version compatibility**: Some SciPy versions may have issues with NumPy 2.x. Use NumPy 1.x if encountering import errors.

2. **Missing PIL dependency**: Some I/O tests may require Pillow, which is not in core dependencies.

## Adding New Tests

When adding new functionality:

1. Write tests first (TDD approach encouraged)
2. Ensure tests cover edge cases
3. Test both valid and invalid inputs
4. Document expected behavior in test docstrings
5. Update this README if adding new test modules
