"""Tests for the quaternions module."""

import numpy as np
import pytest
from pygnd import quaternions, rotations


class TestQuaternionBasicOperaations:
    """Tests for qu_std function."""

    def test_qu_std(self, test_quaternions):
        """Test that quaternions with positive real part are unchanged."""
        test_quaternions *= 2 * np.random.randint(0, 1, (test_quaternions.shape[0], 1)) - 1
        result = quaternions.qu_std(test_quaternions)
        assert (result[:, 0] >= 0.0).all()

    def test_qu_norm(self, test_quaternions):
        """Test that quaternions with negative real part are negated."""
        test_quaternions_altered = test_quaternions * np.random.rand(test_quaternions.shape[0], 1)
        result = quaternions.qu_norm(test_quaternions_altered)
        assert np.allclose(test_quaternions, result)

    def test_qu_norm_std(self, test_quaternions):
        test_quaternions_alt = test_quaternions * np.random.rand(test_quaternions.shape[0], 1)
        test_quaternions_alt *= 2 * np.random.randint(0, 1, (test_quaternions.shape[0], 1)) - 1
        result = quaternions.qu_norm_std(test_quaternions_alt)
        assert np.allclose(np.linalg.norm(result, axis=-1), 1.0)
        assert (result[:, 0] >= 0.0).all()

    def test_qu_conjugate(self, test_quaternions):
        """Test that conjugate quaternions have the same scalar but opposite signed vector"""
        result = quaternions.qu_conj(test_quaternions)
        assert (result[:, 0] >= 0.0).all()
        assert np.allclose(result[:, 1:], test_quaternions[:, 1:] * -1)

    def test_qu_angle(self, test_quaternions):
        """Test that the rotation angle is correctly extracted."""
        angles = rotations.qu2ax(test_quaternions)[..., -1]
        result = quaternions.qu_angle(test_quaternions)
        assert np.allclose(angles, result)

    def test_qu_axis(self, test_quaternions):
        """Test that the rotation angle is correctly extracted."""
        axes = rotations.qu2ax(test_quaternions)[..., :-1]
        result = quaternions.qu_axis(test_quaternions)
        assert np.allclose(axes, result)


class TestQuaternionMultiplication:
    """Tests for qu_prod and qu_prod_raw functions."""

    def test_qu_prod_raw_identity(self, test_quaternions):
        """Tests multiplication by identity quaternion"""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = quaternions.qu_prod_raw(test_quaternions, identity)
        assert np.allclose(test_quaternions, result)

    def test_qu_prod_raw_associativity(self, test_quaternions):
        """Test that quaternion multiplication is associative."""
        a, b, c = test_quaternions[:3]

        # (a*b)*c
        ab = quaternions.qu_prod_raw(a, b)
        abc_1 = quaternions.qu_prod_raw(ab, c)

        # a*(b*c)
        bc = quaternions.qu_prod_raw(b, c)
        abc_2 = quaternions.qu_prod_raw(a, bc)

        np.testing.assert_allclose(abc_1, abc_2, atol=1e-10)

    def test_qu_prod_conjugate(self):
        """Test multiplication of quaternion by its conjugate."""
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        qu_conj = np.array([0.5, -0.5, -0.5, -0.5])
        result = quaternions.qu_prod_raw(qu, qu_conj)
        # Should give a scalar quaternion
        expected_norm_squared = np.sum(qu**2)
        np.testing.assert_allclose(result[0], expected_norm_squared, atol=1e-10)
        np.testing.assert_allclose(result[1:], [0.0, 0.0, 0.0], atol=1e-10)

    def test_qu_prod_standardizes(self):
        """Test that qu_prod standardizes the result."""
        a = np.array([0.5, 0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternions.qu_prod(a, b)
        # Real part should be non-negative
        assert result[0] >= 0

    def test_qu_prod_axis(self, test_quaternions):
        a = np.array([0.5, 0.5, 0.5, 0.5])
        a = quaternions.qu_norm_std(a)
        axis = rotations.qu2ax(quaternions.qu_prod_raw(test_quaternions, a))[..., :-1]
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        result = quaternions.qu_prod_axis(test_quaternions, a)
        result /= np.linalg.norm(result, axis=-1, keepdims=True)
        assert np.allclose(result, axis)


class TestQuaternionProductRealPart:
    """Tests for qu_prod_pos_real function."""

    def test_identity_product(self):
        """Test real part of product with identity."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternions.qu_prod_pos_real(identity, qu)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_orthogonal_quaternions(self):
        """Test real part of product of orthogonal quaternions."""
        a = np.array([0.0, 1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0, 0.0])
        result = quaternions.qu_prod_pos_real(a, b)
        # Real part should be zero for orthogonal pure quaternions
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_parallel_quaternions(self):
        """Test real part of product of parallel quaternions."""
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternions.qu_prod_pos_real(qu, qu)
        # Should be positive
        assert result > 0

    def test_always_positive(self):
        """Test that result is always positive (absolute value)."""
        # These would produce negative real part without abs
        a = np.array([0.0, 1.0, 0.0, 0.0])
        b = np.array([0.0, -1.0, 0.0, 0.0])
        result = quaternions.qu_prod_pos_real(a, b)
        assert result >= 0

    def test_batch_processing(self):
        """Test batch processing of quaternion pairs."""
        a = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        b = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        result = quaternions.qu_prod_pos_real(a, b)
        assert result.shape == (2,)
        assert np.all(result >= 0)


class TestQuaternionTripleProduct:
    """Tests for qu_triple_prod_pos_real function."""

    def test_triple_identity_product(self):
        """Test triple product with identity quaternions."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternions.qu_triple_prod_pos_real(identity, identity, qu)
        # Should be the same as single product real part
        expected = quaternions.qu_prod_pos_real(identity, qu)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_triple_product_consistency(self):
        """Test that triple product equals nested double products."""
        a = np.array([0.5, 0.5, 0.5, 0.5])
        b = np.array([0.7071, 0.7071, 0.0, 0.0])
        c = np.array([0.6, 0.0, 0.8, 0.0])

        result = quaternions.qu_triple_prod_pos_real(a, b, c)
        # Manual computation: a*(b*c)
        bc = quaternions.qu_prod(b, c)
        expected = quaternions.qu_prod_pos_real(a, bc)

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_always_positive(self):
        """Test that result is always positive."""
        a = np.array([0.0, 1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 0.0, 1.0])
        result = quaternions.qu_triple_prod_pos_real(a, b, c)
        assert result >= 0

    def test_batch_processing(self):
        """Test batch processing of quaternion triples."""
        a = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        b = np.array([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.8, 0.0]])
        result = quaternions.qu_triple_prod_pos_real(a, b, c)
        assert result.shape == (2,)
        assert np.all(result >= 0)


class TestQuaternionMisorientations:
    """Tests for calculating misorientations from quaternions"""

    def test_qu_misorientation_identity(self, test_quaternions):
        """Test zero misorientation"""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q_mis = quaternions.qu_misorientation(test_quaternions, identity)
        assert np.allclose(q_mis, test_quaternions)


class TestRotating:
    """Tests using quaternions to rotate vectors"""

    def test_qu_apply_round_trip(self, test_quaternions):
        v = np.array([0.0, 0.0, 1.0])
        v_i = quaternions.qu_apply(test_quaternions, v)
        v_o = quaternions.qu_apply(quaternions.qu_conj(test_quaternions), v_i)
        assert np.allclose(v, v_o)

    def test_qu_apply_misorientation(self, test_quaternions):
        return

    def test_qu_rotate_sets_sphere(self):
        return


class TestSymmetry:
    """Tests for ensuring symmetry operators act correctly"""

    def test_laue_elements(self):
        num_sym = [1, 2, 4, 4, 8, 3, 6, 6, 12, 12, 24]
        for i in range(1, 12):
            s = quaternions.laue_elements(i)
            assert s.shape[0] == num_sym[i - 1], f"Laue group {i} did not load correctly."

    def test_symmetrize(self):
        allowed_misorientation_angles = [
            [0.0],
            [0.0, 180.0],
            [0.0, 180.0],
            [0.0, 90.0, 180.0],
            [0.0, 90.0, 180.0],
            [0.0, 120.0],
            [0.0, 120.0, 180.0],
            [0.0, 60.0, 120.0, 180.0],
            [0.0, 60.0, 120.0, 180.0],
            [0.0, 120.0, 180.0],
            [0.0, 90.0, 120.0, 180.0],
        ]
        a = np.random.rand(1, 4)
        a = quaternions.qu_norm_std(a)
        for i in range(1, 12):
            a_sym = quaternions.symmetrize(a, i)
            s = quaternions.laue_elements(i)
            assert (
                s.shape[0] == a_sym.shape[1]
            ), f"Laue group {i} failed to create symmetric equivalents"

            mis = np.unique(
                np.rad2deg(quaternions.qu_angle(quaternions.qu_misorientation(a, a_sym)))
            )
            allowed_mis = allowed_misorientation_angles[i - 1]
            assert all([np.isclose(m, allowed_mis).any() for m in mis])

    def test_ori_to_fz_laue(self):
        return
