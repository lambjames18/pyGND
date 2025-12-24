"""Tests for rotation representation conversions.

This module tests the round-trip conversion accuracy between different rotation
representations: quaternions (qu), axis-angle (ax), cubochoric (cu), homochoric (ho),
orientation matrix (om), Euler angles (eu), and Rodrigues vectors (ro).
"""

import numpy as np
import pytest
from pygnd import rotations, quaternions


def quaternion_misorientation_degrees(qu1, qu2):
    """Calculate misorientation angle between two quaternions in degrees.

    Args:
        qu1: First quaternion(s) of shape (..., 4)
        qu2: Second quaternion(s) of shape (..., 4)

    Returns:
        Misorientation angle in degrees
    """
    product = quaternions.qu_prod_raw(qu1, quaternions.qu_conj(qu2))
    product_norm = quaternions.qu_norm(product)
    # Clamp to avoid numerical issues with arccos
    cos_angle = np.clip(product_norm[..., 0], -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(np.abs(cos_angle))
    return np.degrees(angle_rad)


class TestQuaternionRoundTrip:
    """Test round-trip conversions starting from quaternions."""

    def test_qu_ax_qu(self, test_quaternions):
        """Test quaternion -> axis-angle -> quaternion conversion."""
        qu_start = test_quaternions
        ax = rotations.qu2ax(qu_start)
        qu_end = rotations.ax2qu(ax)
        q_mis = quaternions.qu_misorientation(qu_start, qu_end)
        error = np.rad2deg(quaternions.qu_angle(q_mis))
        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_qu_cu_qu(self, test_quaternions):
        """Test quaternion -> cubochoric -> quaternion conversion."""
        qu = test_quaternions
        qu1 = rotations.cu2qu(rotations.qu2cu(qu))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_qu_ho_qu(self, test_quaternions):
        """Test quaternion -> homochoric -> quaternion conversion."""
        qu = test_quaternions
        qu1 = rotations.ho2qu(rotations.qu2ho(qu))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_qu_om_qu(self, test_quaternions):
        """Test quaternion -> orientation matrix -> quaternion conversion."""
        qu = test_quaternions
        qu1 = rotations.om2qu(rotations.qu2om(qu))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_qu_eu_qu(self, test_quaternions):
        """Test quaternion -> Euler angles -> quaternion conversion."""
        qu = test_quaternions
        qu1 = rotations.eu2qu(rotations.qu2eu(qu))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_qu_ro_qu(self, test_quaternions):
        """Test quaternion -> Rodrigues -> quaternion conversion."""
        qu = test_quaternions
        qu1 = rotations.ro2qu(rotations.qu2ro(qu))
        error = quaternion_misorientation_degrees(qu, qu1)

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"


class TestEulerAnglesRoundTrip:
    """Test round-trip conversions starting from Euler angles."""

    @pytest.fixture
    def test_euler_angles(self, test_quaternions):
        """Generate test Euler angles from quaternions."""
        return rotations.qu2eu(test_quaternions)

    def test_eu_ax_eu(self, test_euler_angles, test_quaternions):
        """Test Euler angles -> axis-angle -> Euler angles conversion."""
        eu = test_euler_angles
        qu = test_quaternions
        qu1 = rotations.eu2qu(rotations.ax2eu(rotations.eu2ax(eu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_eu_cu_eu(self, test_euler_angles, test_quaternions):
        """Test Euler angles -> cubochoric -> Euler angles conversion."""
        eu = test_euler_angles
        qu = test_quaternions
        qu1 = rotations.eu2qu(rotations.cu2eu(rotations.eu2cu(eu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_eu_ho_eu(self, test_euler_angles, test_quaternions):
        """Test Euler angles -> homochoric -> Euler angles conversion."""
        eu = test_euler_angles
        qu = test_quaternions
        qu1 = rotations.eu2qu(rotations.ho2eu(rotations.eu2ho(eu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_eu_om_eu(self, test_euler_angles, test_quaternions):
        """Test Euler angles -> orientation matrix -> Euler angles conversion."""
        eu = test_euler_angles
        qu = test_quaternions
        qu1 = rotations.eu2qu(rotations.om2eu(rotations.eu2om(eu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_eu_ro_eu(self, test_euler_angles, test_quaternions):
        """Test Euler angles -> Rodrigues -> Euler angles conversion."""
        eu = test_euler_angles
        qu = test_quaternions
        qu1 = rotations.eu2qu(rotations.ro2eu(rotations.eu2ro(eu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"


class TestAxisAngleRoundTrip:
    """Test round-trip conversions starting from axis-angle representation."""

    @pytest.fixture
    def test_axis_angle(self, test_quaternions):
        """Generate test axis-angle pairs from quaternions."""
        return rotations.qu2ax(test_quaternions)

    def test_ax_cu_ax(self, test_axis_angle, test_quaternions):
        """Test axis-angle -> cubochoric -> axis-angle conversion."""
        ax = test_axis_angle
        qu = test_quaternions
        qu1 = rotations.ax2qu(rotations.cu2ax(rotations.ax2cu(ax)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_ax_ho_ax(self, test_axis_angle, test_quaternions):
        """Test axis-angle -> homochoric -> axis-angle conversion."""
        ax = test_axis_angle
        qu = test_quaternions
        qu1 = rotations.ax2qu(rotations.ho2ax(rotations.ax2ho(ax)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_ax_om_ax(self, test_axis_angle, test_quaternions):
        """Test axis-angle -> orientation matrix -> axis-angle conversion."""
        ax = test_axis_angle
        qu = test_quaternions
        qu1 = rotations.ax2qu(rotations.om2ax(rotations.ax2om(ax)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_ax_ro_ax(self, test_axis_angle, test_quaternions):
        """Test axis-angle -> Rodrigues -> axis-angle conversion."""
        ax = test_axis_angle
        qu = test_quaternions
        qu1 = rotations.ax2qu(rotations.ro2ax(rotations.ax2ro(ax)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"


class TestCubochoricRoundTrip:
    """Test round-trip conversions starting from cubochoric representation."""

    @pytest.fixture
    def test_cubochoric(self, test_quaternions):
        """Generate test cubochoric vectors from quaternions."""
        return rotations.qu2cu(test_quaternions)

    def test_cu_ho_cu(self, test_cubochoric, test_quaternions):
        """Test cubochoric -> homochoric -> cubochoric conversion."""
        cu = test_cubochoric
        qu = test_quaternions
        qu1 = rotations.cu2qu(rotations.ho2cu(rotations.cu2ho(cu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_cu_om_cu(self, test_cubochoric, test_quaternions):
        """Test cubochoric -> orientation matrix -> cubochoric conversion."""
        cu = test_cubochoric
        qu = test_quaternions
        qu1 = rotations.cu2qu(rotations.om2cu(rotations.cu2om(cu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_cu_ro_cu(self, test_cubochoric, test_quaternions):
        """Test cubochoric -> Rodrigues -> cubochoric conversion."""
        cu = test_cubochoric
        qu = test_quaternions
        qu1 = rotations.cu2qu(rotations.ro2cu(rotations.cu2ro(cu)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"


class TestHomochoricRoundTrip:
    """Test round-trip conversions starting from homochoric representation."""

    @pytest.fixture
    def test_homochoric(self, test_quaternions):
        """Generate test homochoric vectors from quaternions."""
        return rotations.qu2ho(test_quaternions)

    def test_ho_om_ho(self, test_homochoric, test_quaternions):
        """Test homochoric -> orientation matrix -> homochoric conversion."""
        ho = test_homochoric
        qu = test_quaternions
        qu1 = rotations.ho2qu(rotations.om2ho(rotations.ho2om(ho)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"

    def test_ho_ro_ho(self, test_homochoric, test_quaternions):
        """Test homochoric -> Rodrigues -> homochoric conversion."""
        ho = test_homochoric
        qu = test_quaternions
        qu1 = rotations.ho2qu(rotations.ro2ho(rotations.ho2ro(ho)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"


class TestOrientationMatrixRoundTrip:
    """Test round-trip conversions starting from orientation matrix representation."""

    @pytest.fixture
    def test_orientation_matrix(self, test_quaternions):
        """Generate test orientation matrices from quaternions."""
        return rotations.qu2om(test_quaternions)

    def test_om_ro_om(self, test_orientation_matrix, test_quaternions):
        """Test orientation matrix -> Rodrigues -> orientation matrix conversion."""
        om = test_orientation_matrix
        qu = test_quaternions
        qu1 = rotations.om2qu(rotations.ro2om(rotations.om2ro(om)))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        assert error.max() < 1e-6, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-8, f"Mean error: {error.mean()} degrees"


class TestFloat32Precision:
    """Test conversions with float32 precision."""

    @pytest.fixture
    def test_quaternions_float32(self):
        """Generate float32 test quaternions."""
        np.random.seed(1)
        n = int(1e6)

        # Generate random quaternions
        qu = np.random.random((n, 4)).astype(np.float32)
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
                    np.float32
                ),  # Arbitrary
            ]
        )
        return qu

    def test_qu_ax_qu_float32(self, test_quaternions_float32):
        """Test quaternion -> axis-angle -> quaternion with float32."""
        qu = test_quaternions_float32
        qu1 = rotations.ax2qu(rotations.qu2ax(qu))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        # Float32 has less precision, so we allow larger errors
        assert error.max() < 1e-4, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-6, f"Mean error: {error.mean()} degrees"

    def test_qu_om_qu_float32(self, test_quaternions_float32):
        """Test quaternion -> orientation matrix -> quaternion with float32."""
        qu = test_quaternions_float32
        qu1 = rotations.om2qu(rotations.qu2om(qu))
        q_mis = quaternions.qu_misorientation(qu, qu1)
        error = np.rad2deg(quaternions.qu_angle(q_mis))

        # Float32 has less precision
        assert error.max() < 1e-4, f"Max error: {error.max()} degrees"
        assert error.mean() < 1e-6, f"Mean error: {error.mean()} degrees"
