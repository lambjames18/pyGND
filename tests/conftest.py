"""Pytest configuration and fixtures for PyGND tests."""

import pytest
import numpy as np
from pathlib import Path

from pygnd import quaternions


@pytest.fixture
def sample_quaternion():
    """Provide a sample normalized quaternion."""
    return np.array([0.5, 0.5, 0.5, 0.5])


@pytest.fixture
def identity_quaternion():
    """Provide an identity quaternion."""
    return np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def sample_euler_angles():
    """Provide sample Euler angles in radians."""
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    np.random.seed(None)  # Reset seed after test


@pytest.fixture
def dream3d_file():
    """Path to a sample DREAM.3D file for testing."""
    return "./demo_data/CoNi.dream3d"


@pytest.fixture
def xdmf_file():
    """Path to a sample XDMF file for testing."""
    return "./demo_data/CoNi.xdmf"


@pytest.fixture
def ang_file():
    """Path to a sample .ang file for testing."""
    return Path("./demo_data/CoNi.ang")


@pytest.fixture
def grain_data_file():
    """Path to a sample grain data file for testing."""
    return "./demo_data/CoNi_grain_data.txt"


@pytest.fixture
def test_quaternions():
    """Generate a set of test quaternions including edge cases."""
    np.random.seed(1)
    n = int(1e6)

    # Generate random quaternions
    qu = np.random.random((n, 4)).astype(np.float64)
    qu = quaternions.qu_norm(qu)

    # Add specific edge cases
    qu = np.vstack(
        [
            qu,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],  # Identity
                    [0.0, 0.0, 0.0, -1.0],  # 180-degree rotation
                    [0.0, 0.4264014489670335, -0.6396021436482270, -0.6396021436482270],
                ]
            ).astype(
                np.float64
            ),  # Arbitrary
        ]
    )
    return qu
