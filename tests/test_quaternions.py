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
        """Misorientation relative to identity should rotate points identically to q itself."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        mis = quaternions.qu_misorientation(test_quaternions, identity)
        v_direct = quaternions.qu_apply(test_quaternions, v)
        v_via_mis = quaternions.qu_apply(mis, v)
        assert np.allclose(v_direct, v_via_mis)

    def test_qu_rotate_sets_sphere(self):
        """Returned quaternions should rotate points_start exactly onto points_finish."""
        p0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        q = quaternions.qu_rotate_sets_sphere(p0, p1)
        p1_check = quaternions.qu_apply(q, p0)
        assert np.allclose(p1_check, p1, atol=1e-6)
        assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)

    def test_qu_rotate_sets_sphere_identical_points(self):
        """Identical start/finish points are numerically unstable and should yield identity."""
        p_same = np.array([[1.0, 0.0, 0.0]])
        q_same = quaternions.qu_rotate_sets_sphere(p_same, p_same)
        assert np.allclose(q_same, [[1.0, 0.0, 0.0, 0.0]])


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
        """Triclinic (identity-only group) should leave orientations unchanged."""
        np.random.seed(7)
        q = quaternions.qu_norm_std(np.random.rand(5, 4))
        result = quaternions.ori_to_fz_laue(q, laue_id=1)
        assert np.allclose(result, q)

    def test_ori_to_fz_laue_max_real_part(self):
        """The FZ representative should have the largest |real part| among all
        symmetric equivalents, by construction."""
        np.random.seed(8)
        q = quaternions.qu_norm_std(np.random.rand(6, 4))
        laue_id = 11
        laue_group = quaternions.laue_elements(laue_id)
        result = quaternions.ori_to_fz_laue(q, laue_id=laue_id)
        equivalents = quaternions.qu_prod(q[:, None], laue_group[None])
        max_real = np.abs(equivalents[..., 0]).max(axis=-1)
        assert np.allclose(np.abs(result[..., 0]), max_real, atol=1e-6)


class TestDisorientation:
    """Tests for qu_disorientation and qu_disorientation_directional."""

    def test_qu_disorientation_no_symmetry_matches_misorientation(self):
        """With the triclinic (identity-only) Laue group, disorientation should
        reduce to the plain misorientation quaternion."""
        np.random.seed(10)
        q1 = quaternions.qu_norm_std(np.random.rand(5, 4))
        q2 = quaternions.qu_norm_std(np.random.rand(5, 4))
        expected = quaternions.qu_std(quaternions.qu_misorientation(q1, q2))
        result = quaternions.qu_disorientation(q1, q2, laue_id_1=1, laue_id_2=1)
        assert np.allclose(result, expected, atol=1e-6)

    def test_qu_disorientation_identical_orientations_is_identity(self):
        """Disorientation between identical orientations should be the identity
        quaternion regardless of symmetry."""
        np.random.seed(11)
        q = quaternions.qu_norm_std(np.random.rand(4, 4))
        for laue_id in (1, 9, 11):
            result = quaternions.qu_disorientation(q, q, laue_id, laue_id)
            assert np.allclose(np.abs(result[..., 0]), 1.0, atol=1e-6)

    def test_qu_disorientation_reduces_angle_relative_to_misorientation(self):
        """Applying symmetry should never increase the disorientation angle
        relative to the raw misorientation."""
        np.random.seed(12)
        q1 = quaternions.qu_norm_std(np.random.rand(20, 4))
        q2 = quaternions.qu_norm_std(np.random.rand(20, 4))
        mis_angle = quaternions.qu_angle(quaternions.qu_misorientation(q1, q2))
        dis = quaternions.qu_disorientation(q1, q2, laue_id_1=11, laue_id_2=11)
        dis_angle = quaternions.qu_angle(dis)
        assert np.all(dis_angle <= mis_angle + 1e-6)

    def test_qu_disorientation_mismatched_shape_raises(self):
        """Mismatched quats1/quats2 shapes should raise a ValueError. This is a
        regression test: a prior bug compared quats2's shape to itself (always
        true), so mismatched shapes silently passed instead of raising."""
        q1 = quaternions.qu_norm_std(np.random.rand(3, 4))
        q2 = quaternions.qu_norm_std(np.random.rand(5, 4))
        with pytest.raises(ValueError):
            quaternions.qu_disorientation(q1, q2, laue_id_1=1, laue_id_2=1)

    def test_qu_disorientation_invalid_dtype_raises(self):
        q1 = quaternions.qu_norm_std(np.random.rand(3, 4)).astype(np.int32)
        q2 = quaternions.qu_norm_std(np.random.rand(3, 4))
        with pytest.raises(ValueError, match="float32 or float64"):
            quaternions.qu_disorientation(q1, q2, laue_id_1=1, laue_id_2=1)

    def test_qu_disorientation_non_naive_matches_naive_angle(self):
        """The non-naive path additionally restricts the axis to the fundamental
        sector, but the disorientation angle itself should match the naive path."""
        np.random.seed(22)
        q1 = quaternions.qu_norm_std(np.random.rand(6, 4))
        q2 = quaternions.qu_norm_std(np.random.rand(6, 4))
        naive = quaternions.qu_disorientation(q1, q2, laue_id_1=11, laue_id_2=11, naive=True)
        non_naive = quaternions.qu_disorientation(
            q1, q2, laue_id_1=11, laue_id_2=11, naive=False
        )
        assert np.allclose(
            quaternions.qu_angle(naive), quaternions.qu_angle(non_naive), atol=1e-5
        )

    def test_qu_disorientation_directional_no_symmetry_matches_misorientation(self):
        np.random.seed(13)
        q1 = quaternions.qu_norm_std(np.random.rand(5, 4))
        q2 = quaternions.qu_norm_std(np.random.rand(5, 4))
        expected = quaternions.qu_std(quaternions.qu_misorientation(q1, q2))
        result = quaternions.qu_disorientation_directional(q1, q2, laue_id=1)
        assert np.allclose(result, expected, atol=1e-6)

    def test_qu_disorientation_directional_mismatched_shape_raises(self):
        """Regression test for the same shape-validation bug as above."""
        q1 = quaternions.qu_norm_std(np.random.rand(3, 4))
        q2 = quaternions.qu_norm_std(np.random.rand(5, 4))
        with pytest.raises(ValueError):
            quaternions.qu_disorientation_directional(q1, q2, laue_id=1)

    def test_qu_disorientation_directional_recognizes_symmetric_equivalents(self):
        """Regression test: quats1 was never reshaped before broadcasting against
        the symmetrized quats2, so for batches (N > 1) each sample was compared
        against every other sample's symmetric equivalents instead of just its
        own, giving nonzero "disorientation" between genuinely symmetric-
        equivalent pairs."""
        np.random.seed(21)
        laue_id = 9
        q1 = quaternions.qu_norm_std(np.random.rand(5, 4))
        laue_group = quaternions.laue_elements(laue_id)
        idx = np.random.randint(0, laue_group.shape[0], size=5)
        # symmetric equivalents use the pre-multiply (S * q) convention, matching
        # symmetrize() and qu_disorientation_directional's own internal convention
        q2 = quaternions.qu_prod(laue_group[idx], q1)
        result = quaternions.qu_disorientation_directional(q1, q2, laue_id=laue_id)
        assert np.allclose(np.abs(result[..., 0]), 1.0, atol=1e-6)


class TestSlerpAndLog:
    """Tests for qu_slerp and qu_log."""

    def test_qu_slerp_endpoints(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = quaternions.qu_norm_std(np.array([0.70710678, 0.70710678, 0.0, 0.0]))
        assert np.allclose(quaternions.qu_slerp(a, b, 0.0), a, atol=1e-6)
        assert np.allclose(quaternions.qu_slerp(a, b, 1.0), b, atol=1e-6)

    def test_qu_slerp_unit_norm_and_batch(self):
        """Slerp should preserve unit norm and broadcast correctly over a batch."""
        np.random.seed(14)
        a = quaternions.qu_norm_std(np.random.rand(10, 4))
        b = quaternions.qu_norm_std(np.random.rand(10, 4))
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            mid = quaternions.qu_slerp(a, b, t)
            assert mid.shape == a.shape
            assert np.allclose(np.linalg.norm(mid, axis=-1), 1.0, atol=1e-6)

    def test_qu_log_identity_is_zero(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(quaternions.qu_log(identity), [0.0, 0.0, 0.0])

    def test_qu_log_matches_qu_angle(self, test_quaternions):
        """2 * ||log(q)|| should equal the rotation angle from qu_angle."""
        logs = quaternions.qu_log(test_quaternions)
        angle_from_log = 2 * np.linalg.norm(logs, axis=-1)
        angle_direct = quaternions.qu_angle(test_quaternions)
        assert np.allclose(angle_from_log, angle_direct, atol=1e-5)


class TestQuAvg:
    """Tests for qu_avg."""

    def test_qu_avg_identical_quaternions(self):
        """Averaging N copies of the same quaternion should recover it."""
        np.random.seed(15)
        q = quaternions.qu_norm_std(np.random.rand(1, 4))
        q_rep = np.repeat(q, 8, axis=0)
        avg = quaternions.qu_avg(q_rep, laue_id=1)
        assert np.allclose(np.abs(avg), np.abs(q[0]), atol=1e-6)

    def test_qu_avg_with_symmetry_variants(self):
        """When qn are the same true orientation expressed via arbitrary (and
        differing) symmetric representatives, qu_avg should recognize the
        equivalence (via the reference q0) and recover the true orientation."""
        np.random.seed(16)
        laue_id = 9
        q_true = quaternions.qu_norm_std(np.random.rand(1, 4))[0]
        laue_group = quaternions.laue_elements(laue_id)
        idx = np.random.randint(0, laue_group.shape[0], size=10)
        # pre-multiply (S * q) convention, matching symmetrize()'s convention
        variants = quaternions.qu_prod(laue_group[idx], np.tile(q_true, (10, 1)))
        q_set = np.vstack([q_true[None, :], variants])
        avg = quaternions.qu_avg(q_set, laue_id=laue_id)
        avg = avg / np.linalg.norm(avg)
        assert np.isclose(np.dot(avg, q_true), 1.0, atol=1e-4)


class TestDisorientationStatistics:
    """Tests for get_Q_tensor, get_preferred_rotation_axis, and
    get_sign_carrying_disorientation_angle."""

    @staticmethod
    def _axis_angle_quaternions(axis, angles):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = np.asarray(angles, dtype=float) / 2.0
        w = np.cos(half)
        xyz = axis[None, :] * np.sin(half)[:, None]
        return np.column_stack([w, xyz])

    def test_get_q_tensor_symmetric(self):
        np.random.seed(20)
        q_dis = quaternions.qu_norm_std(np.random.rand(50, 4))
        Q = quaternions.get_Q_tensor(q_dis)
        assert Q.shape == (3, 3)
        assert np.allclose(Q, Q.T, atol=1e-10)

    def test_get_preferred_rotation_axis_matches_common_axis(self):
        """When all disorientations share a common rotation axis, the preferred
        axis (dominant eigenvector of the Q tensor) should align with it."""
        axis = np.array([0.0, 0.0, 1.0])
        angles = np.linspace(0.1, 1.5, 30)
        q_dis = self._axis_angle_quaternions(axis, angles)
        r_star = quaternions.get_preferred_rotation_axis(q_dis)
        assert np.isclose(np.linalg.norm(r_star), 1.0, atol=1e-6)
        assert np.isclose(np.abs(np.dot(r_star, axis)), 1.0, atol=1e-4)

    def test_get_sign_carrying_disorientation_angle_identity_is_zero(self):
        identity = np.array([[1.0, 0.0, 0.0, 0.0]])
        angles = quaternions.get_sign_carrying_disorientation_angle(
            identity, r_star=np.array([0.0, 0.0, 1.0])
        )
        assert np.allclose(angles, 0.0)

    def test_get_sign_carrying_disorientation_angle_tracks_half_angle(self):
        """For disorientations about the reference axis itself, the sign-carrying
        angle should equal the signed half-angle of rotation."""
        axis = np.array([0.0, 0.0, 1.0])
        angles_in = np.linspace(-1.0, 1.0, 21)
        angles_in = angles_in[np.abs(angles_in) > 1e-3]
        q_dis = self._axis_angle_quaternions(axis, angles_in)
        angles_out = quaternions.get_sign_carrying_disorientation_angle(q_dis, r_star=axis)
        assert np.all(np.isfinite(angles_out))
        assert np.allclose(angles_out, angles_in / 2.0, atol=1e-6)
