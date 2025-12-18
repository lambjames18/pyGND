"""Tests for the quaternions module."""

import numpy as np
import pytest
from pygnd import quaternions


class TestQuaternionStandardization:
    """Tests for qu_std function."""

    def test_positive_real_part_unchanged(self):
        """Test that quaternions with positive real part are unchanged."""
        qu = np.array([1.0, 0.0, 0.0, 0.0])
        result = quaternions.qu_std(qu)
        np.testing.assert_array_equal(result, qu)

    def test_negative_real_part_negated(self):
        """Test that quaternions with negative real part are negated."""
        qu = np.array([-1.0, 0.0, 0.0, 0.0])
        result = quaternions.qu_std(qu)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_zero_real_part(self):
        """Test that quaternions with zero real part are unchanged."""
        qu = np.array([0.0, 1.0, 0.0, 0.0])
        result = quaternions.qu_std(qu)
        np.testing.assert_array_equal(result, qu)

    def test_batch_quaternions(self):
        """Test batch processing of multiple quaternions."""
        qu = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
        result = quaternions.qu_std(qu)
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(result, expected)


class TestQuaternionNormalization:
    """Tests for qu_norm function."""

    def test_unit_quaternion_unchanged(self):
        """Test that unit quaternions are unchanged."""
        qu = np.array([1.0, 0.0, 0.0, 0.0])
        result = quaternions.qu_norm(qu)
        np.testing.assert_allclose(result, qu, atol=1e-10)

    def test_normalization(self):
        """Test that quaternions are normalized to unit length."""
        qu = np.array([2.0, 0.0, 0.0, 0.0])
        result = quaternions.qu_norm(qu)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_zero_quaternion(self):
        """Test that zero quaternions are handled correctly."""
        qu = np.array([0.0, 0.0, 0.0, 0.0])
        result = quaternions.qu_norm(qu)
        np.testing.assert_array_equal(result, 0.0)

    def test_arbitrary_quaternion(self):
        """Test normalization of arbitrary quaternion."""
        qu = np.array([1.0, 2.0, 3.0, 4.0])
        result = quaternions.qu_norm(qu)
        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_batch_quaternions(self):
        """Test batch processing of multiple quaternions."""
        qu = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        result = quaternions.qu_norm(qu)
        norms = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)


class TestQuaternionMultiplication:
    """Tests for qu_prod and qu_prod_raw functions."""

    def test_identity_multiplication(self):
        """Test multiplication by identity quaternion."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        qu = np.array([0.5, 0.5, 0.5, 0.5])
        result = quaternions.qu_prod(identity, qu)
        # Normalize both for comparison
        qu_normalized = quaternions.qu_std(qu)
        np.testing.assert_allclose(result, qu_normalized, atol=1e-10)

    def test_multiplication_associativity(self):
        """Test that quaternion multiplication is associative."""
        a = np.array([0.5, 0.5, 0.5, 0.5])
        b = np.array([0.7071, 0.7071, 0.0, 0.0])
        c = np.array([0.6, 0.0, 0.8, 0.0])

        # (a*b)*c
        ab = quaternions.qu_prod_raw(a, b)
        abc_1 = quaternions.qu_prod_raw(ab, c)

        # a*(b*c)
        bc = quaternions.qu_prod_raw(b, c)
        abc_2 = quaternions.qu_prod_raw(a, bc)

        np.testing.assert_allclose(abc_1, abc_2, atol=1e-10)

    def test_conjugate_multiplication(self):
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

    def test_batch_multiplication(self):
        """Test batch quaternion multiplication."""
        a = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        b = np.array([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
        result = quaternions.qu_prod(a, b)
        assert result.shape == (2, 4)


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


class TestQuaternionProperties:
    """General tests for quaternion operations."""

    def test_norm_preservation_in_product(self):
        """Test that quaternion product preserves norm (for unit quaternions)."""
        a = quaternions.qu_norm(np.array([1.0, 1.0, 1.0, 1.0]))
        b = quaternions.qu_norm(np.array([2.0, 1.0, 0.0, 1.0]))
        product = quaternions.qu_prod_raw(a, b)
        product_norm = np.linalg.norm(product)
        np.testing.assert_allclose(product_norm, 1.0, atol=1e-10)

    def test_standardization_idempotent(self):
        """Test that standardization is idempotent."""
        qu = np.array([-0.5, 0.5, 0.5, 0.5])
        std1 = quaternions.qu_std(qu)
        std2 = quaternions.qu_std(std1)
        np.testing.assert_array_equal(std1, std2)

    def test_normalization_idempotent(self):
        """Test that normalization is idempotent for non-zero quaternions."""
        qu = np.array([1.0, 2.0, 3.0, 4.0])
        norm1 = quaternions.qu_norm(qu)
        norm2 = quaternions.qu_norm(norm1)
        np.testing.assert_allclose(norm1, norm2, atol=1e-10)
