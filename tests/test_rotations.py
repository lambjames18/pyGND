"""Tests for the rotations module."""

import numpy as np
import pytest
from pygnd import rotations


class TestQuaternionToOrientationMatrix:
    """Tests for qu2om function."""

    def test_identity_quaternion(self):
        """Test that identity quaternion produces identity matrix."""
        qu = np.array([1.0, 0.0, 0.0, 0.0])
        om = rotations.qu2om(qu)
        expected = np.eye(3)
        np.testing.assert_allclose(om, expected, atol=1e-10)

    def test_90_degree_rotation_z(self):
        """Test 90 degree rotation around z-axis."""
        # Quaternion for 90 degree rotation around z-axis
        qu = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        om = rotations.qu2om(qu)
        # Expected rotation matrix for 90 degrees around z
        expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(om, expected, atol=1e-10)

    def test_batch_quaternions(self):
        """Test batch processing of multiple quaternions."""
        qu = np.array([[1.0, 0.0, 0.0, 0.0], [np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2]])
        om = rotations.qu2om(qu)
        assert om.shape == (2, 3, 3)

    def test_orthogonality(self):
        """Test that output matrix is orthogonal (R^T * R = I)."""
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        om = rotations.qu2om(qu)
        product = om @ om.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)


class TestQuaternionToEulerAngles:
    """Tests for qu2eu function."""

    def test_identity_quaternion(self):
        """Test that identity quaternion produces zero Euler angles."""
        qu = np.array([1.0, 0.0, 0.0, 0.0])
        eu = rotations.qu2eu(qu)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(eu, expected, atol=1e-10)

    def test_euler_angle_range(self):
        """Test that Euler angles are in valid range."""
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        eu = rotations.qu2eu(qu)
        # phi1 and phi2 should be in [0, 2*pi]
        assert 0 <= eu[0] <= 2 * np.pi
        assert 0 <= eu[2] <= 2 * np.pi
        # Phi should be in [0, pi]
        assert 0 <= eu[1] <= np.pi

    def test_batch_quaternions(self):
        """Test batch processing of multiple quaternions."""
        qu = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        eu = rotations.qu2eu(qu)
        assert eu.shape == (2, 3)


class TestQuaternionToAxisAngle:
    """Tests for qu2ax function."""

    def test_identity_quaternion(self):
        """Test that identity quaternion produces zero rotation."""
        qu = np.array([1.0, 0.0, 0.0, 0.0])
        ax = rotations.qu2ax(qu)
        # Angle should be zero
        np.testing.assert_allclose(ax[3], 0.0, atol=1e-10)

    def test_180_degree_rotation(self):
        """Test 180 degree rotation."""
        # 180 degree rotation around x-axis
        qu = np.array([0.0, 1.0, 0.0, 0.0])
        ax = rotations.qu2ax(qu)
        # Angle should be pi
        np.testing.assert_allclose(ax[3], np.pi, atol=1e-10)
        # Axis should be normalized
        axis_norm = np.linalg.norm(ax[:3])
        np.testing.assert_allclose(axis_norm, 1.0, atol=1e-10)

    def test_axis_normalization(self):
        """Test that rotation axis is normalized."""
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        ax = rotations.qu2ax(qu)
        axis_norm = np.linalg.norm(ax[:3])
        np.testing.assert_allclose(axis_norm, 1.0, atol=1e-10)

    def test_batch_quaternions(self):
        """Test batch processing of multiple quaternions."""
        qu = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        ax = rotations.qu2ax(qu)
        assert ax.shape == (2, 4)


class TestEpsijk:
    """Test the epsijk parameter."""

    def test_epsijk_value(self):
        """Test that epsijk is set to expected value."""
        assert rotations.epsijk == 1


class TestConversionConsistency:
    """Test consistency between different conversion functions."""

    def test_qu2om_determinant(self):
        """Test that all rotation matrices have determinant of 1."""
        quaternions = [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0],
        ]
        for qu in quaternions:
            om = rotations.qu2om(np.array(qu))
            det = np.linalg.det(om)
            np.testing.assert_allclose(det, 1.0, atol=1e-10)

    def test_multiple_conversions(self):
        """Test that conversions work for various quaternions."""
        test_quaternions = [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ]

        for qu in test_quaternions:
            qu_arr = np.array(qu)
            # All conversions should execute without error
            om = rotations.qu2om(qu_arr)
            eu = rotations.qu2eu(qu_arr)
            ax = rotations.qu2ax(qu_arr)

            assert om.shape == (3, 3)
            assert eu.shape == (3,)
            assert ax.shape == (4,)
