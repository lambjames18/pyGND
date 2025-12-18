"""Pytest configuration and fixtures for PyGND tests."""

import pytest
import numpy as np
from pathlib import Path


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
