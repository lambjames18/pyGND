"""Tests for the core module."""

import numpy as np
import pytest
from pygnd import core, rotations


class TestGetLinearOperator:
    """Tests for get_linear_operator function."""

    def test_fcc_structure(self):
        """Test that FCC crystal structure returns correct A and B matrices."""
        A, B = core.get_linear_operator(cs=1, slip_systems="all")

        # Check shapes
        assert A.shape[0] == 9  # 9 components of the curvature tensor
        assert A.shape[1] == 18  # 18 slip systems for FCC
        assert B.shape == (18, 9)

    def test_bcc_structure_all_slip_systems(self):
        """Test that BCC crystal structure with all slip systems works."""
        A, B = core.get_linear_operator(cs=2, slip_systems="all")

        # Check shapes
        assert A.shape[0] == 9
        assert A.shape[1] == 52  # All BCC slip systems
        assert B.shape[0] == 52
        assert B.shape[1] == 9

    def test_bcc_structure_screw_110(self):
        """Test BCC with screw+110 slip systems."""
        A, B = core.get_linear_operator(cs=2, slip_systems="screw+110")

        # Check shapes
        assert A.shape[0] == 9
        assert A.shape[1] == 16
        assert B.shape == (16, 9)

    def test_bcc_structure_screw_112(self):
        """Test BCC with screw+112 slip systems."""
        A, B = core.get_linear_operator(cs=2, slip_systems="screw+112")

        assert A.shape[0] == 9
        assert A.shape[1] == 16  # 4 screw + 12 edge on {112}

    def test_bcc_structure_screw_123(self):
        """Test BCC with screw+123 slip systems."""
        A, B = core.get_linear_operator(cs=2, slip_systems="screw+123")

        assert A.shape[0] == 9
        assert A.shape[1] == 28  # 4 screw + 24 edge on {123}

    def test_bcc_structure_screw_110_112(self):
        """Test BCC with screw+110+112 slip systems."""
        A, B = core.get_linear_operator(cs=2, slip_systems="screw+110+112")

        assert A.shape[0] == 9
        assert A.shape[1] == 28

    def test_bcc_structure_screw_110_123(self):
        """Test BCC with screw+110+123 slip systems."""
        A, B = core.get_linear_operator(cs=2, slip_systems="screw+110+123")

        assert A.shape[0] == 9
        assert A.shape[1] == 40

    def test_bcc_structure_screw_112_123(self):
        """Test BCC with screw+112+123 slip systems."""
        A, B = core.get_linear_operator(cs=2, slip_systems="screw+112+123")

        assert A.shape[0] == 9
        assert A.shape[1] == 40  # 8 screw + 36 edge on {112} and {123}

    def test_hcp_structure_all_slip_systems(self):
        """Test that HCP crystal structure with all slip systems works."""
        A, B = core.get_linear_operator(cs=3, slip_systems="all")

        # Check shapes
        assert A.shape[0] == 9
        assert A.shape[1] == 33  # All HCP slip systems
        assert B.shape[0] == 33
        assert B.shape[1] == 9

    def test_hcp_structure_basal(self):
        """Test HCP with basal slip systems."""
        A, B = core.get_linear_operator(cs=3, slip_systems="basal")

        assert A.shape[0] == 9
        assert A.shape[1] == 6  # 3 edge + 3 screw basal
        assert B.shape == (6, 9)

    def test_hcp_structure_prismatic(self):
        """Test HCP with prismatic slip systems."""
        A, B = core.get_linear_operator(cs=3, slip_systems="prismatic")

        assert A.shape[0] == 9
        assert A.shape[1] == 3  # 3 edge
        assert B.shape == (3, 9)

    def test_hcp_structure_pyramidal(self):
        """Test HCP with pyramidal slip systems."""
        A, B = core.get_linear_operator(cs=3, slip_systems="pyramidal")

        assert A.shape[0] == 9
        assert A.shape[1] == 24  # 12 edge + 12 screw pyramidal
        assert B.shape == (24, 9)

    def test_hcp_structure_basal_prismatic(self):
        """Test HCP with basal+prismatic slip systems."""
        A, B = core.get_linear_operator(cs=3, slip_systems="basal+prismatic")

        assert A.shape[0] == 9
        assert A.shape[1] == 9
        assert B.shape == (9, 9)

    def test_hcp_structure_basal_pyramidal(self):
        """Test HCP with basal+pyramidal slip systems."""
        A, B = core.get_linear_operator(cs=3, slip_systems="basal+pyramidal")

        assert A.shape[0] == 9
        assert A.shape[1] == 30
        assert B.shape == (30, 9)

    def test_hcp_structure_prismatic_pyramidal(self):
        """Test HCP with prismatic+pyramidal slip systems."""
        A, B = core.get_linear_operator(cs=3, slip_systems="prismatic+pyramidal")

        assert A.shape[0] == 9
        assert A.shape[1] == 27
        assert B.shape == (27, 9)

    def test_invalid_crystal_structure_type(self):
        """Test that non-integer crystal structure raises ValueError."""
        with pytest.raises(ValueError, match="Crystal structure must be an integer"):
            core.get_linear_operator(cs=1.5, slip_systems="all")

    def test_invalid_crystal_structure_value(self):
        """Test that invalid crystal structure value raises ValueError."""
        with pytest.raises(ValueError, match="Crystal structure must be 1, 2, or 3"):
            core.get_linear_operator(cs=4, slip_systems="all")

    def test_invalid_slip_systems_type(self):
        """Test that non-string slip systems raises ValueError."""
        with pytest.raises(ValueError, match="Slip systems must be a string"):
            core.get_linear_operator(cs=1, slip_systems=123)

    def test_invalid_slip_systems_value(self):
        """Test that invalid slip system string raises ValueError."""
        with pytest.raises(ValueError, match="Slip systems must be"):
            core.get_linear_operator(cs=1, slip_systems="invalid_slip_system")

    def test_slip_systems_case_insensitive(self):
        """Test that slip systems string is case-insensitive."""
        A1, B1 = core.get_linear_operator(cs=2, slip_systems="ALL")
        A2, B2 = core.get_linear_operator(cs=2, slip_systems="all")

        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)

    def test_slip_systems_whitespace_stripped(self):
        """Test that whitespace is stripped from slip systems string."""
        A1, B1 = core.get_linear_operator(cs=2, slip_systems="  all  ")
        A2, B2 = core.get_linear_operator(cs=2, slip_systems="all")

        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)

    def test_matrix_dtypes(self):
        """Test that matrices use correct precision."""
        A, B = core.get_linear_operator(cs=1, slip_systems="all")

        # Both should be float32 based on PRECISION constant
        assert A.dtype == np.float32
        assert B.dtype == np.float32

    def test_a_times_b_near_identity(self):
        """Test that A @ B is close to identity for full-rank underdetermined systems.

        For underdetermined systems (m < n) with full rank, A @ B ≈ I_m should hold.
        For rank-deficient systems, we test the Moore-Penrose property A @ B @ A ≈ A instead.
        """
        structures = [
            (1, "all"),
            (2, "all"),
            (2, "screw+110"),
            (2, "screw+112"),
            (2, "screw+123"),
            (2, "screw+110+112"),
            (2, "screw+110+123"),
            (2, "screw+112+123"),
            (3, "all"),
            (3, "basal"),
            (3, "prismatic"),
            (3, "pyramidal"),
            (3, "basal+prismatic"),
            (3, "basal+pyramidal"),
            (3, "prismatic+pyramidal"),
        ]

        fail_count = 0
        for cs, slip_sys in structures:
            A, B = core.get_linear_operator(cs=cs, slip_systems=slip_sys)
            m, n = A.shape
            rank = np.linalg.matrix_rank(A)

            # For full-rank underdetermined systems, test A @ B ≈ I
            if rank == m and m < n:
                product = A @ B
                error = np.linalg.norm(product - np.eye(m), ord="fro")
                if error >= 1e-6:
                    print(
                        f"Failed A @ B ≈ I check for cs={cs}, slip_systems={slip_sys}: {error}"
                    )
                    fail_count += 1

            # For rank-deficient systems, test the Moore-Penrose property: A @ B @ A ≈ A
            else:
                product = A @ B @ A
                error = np.linalg.norm(product - A, ord="fro") / np.linalg.norm(A, ord="fro")
                if error >= 1e-6:
                    print(
                        f"Failed A @ B @ A ≈ A check for cs={cs}, slip_systems={slip_sys}: {error}"
                    )
                    fail_count += 1

        assert fail_count == 0, "Some pseudo-inverse property checks failed."


class TestPrecisionConstant:
    """Test the PRECISION constant."""

    def test_precision_is_float32(self):
        """Test that PRECISION is set to float32."""
        assert core.PRECISION == np.float32


class TestMatrixProperties:
    """Test mathematical properties of generated matrices."""

    def test_fcc_matrix_rank(self):
        """Test that FCC A matrix has appropriate rank."""
        A, B = core.get_linear_operator(cs=1, slip_systems="all")
        rank = np.linalg.matrix_rank(A)

        # Rank should be at most min(9, 18) = 9
        assert rank <= 9
        # For GND calculations, we expect rank close to 9
        assert rank >= 8

    def test_bcc_matrix_rank(self):
        """Test that BCC A matrix has appropriate rank."""
        A, B = core.get_linear_operator(cs=2, slip_systems="all")
        rank = np.linalg.matrix_rank(A)

        # Rank should be at most 9
        assert rank <= 9
        assert rank >= 8

    def test_hcp_matrix_rank(self):
        """Test that HCP A matrix has appropriate rank."""
        A, B = core.get_linear_operator(cs=3, slip_systems="all")
        rank = np.linalg.matrix_rank(A)

        # Rank should be at most 9
        assert rank <= 9
        assert rank >= 8

    def test_matrix_condition_number_reasonable(self):
        """Test that condition numbers are not too large."""
        structures = [(1, "all"), (2, "all"), (3, "all")]

        for cs, slip_sys in structures:
            A, B = core.get_linear_operator(cs=cs, slip_systems=slip_sys)
            cond = np.linalg.cond(A)

            # Condition number shouldn't be extremely large
            # (though some ill-conditioning is expected in GND calculations)
            assert cond < 1e10, f"Condition number too large for cs={cs}: {cond}"


class TestGetCompleteness:
    """Tests for get_completeness function."""

    def test_single_grain_boundaries_and_interior(self):
        """A single grain filling the volume should have central differences
        (3) in the interior and one-sided differences (1 forward, 2 backward)
        at the outer faces of the volume."""
        ids = np.ones((4, 4, 4), dtype=int)
        c = core.get_completeness(ids)

        assert np.array_equal(c[0, 1, 1], [1, 3, 3])
        assert np.array_equal(c[-1, 1, 1], [2, 3, 3])
        assert np.array_equal(c[1, 0, 1], [3, 1, 3])
        assert np.array_equal(c[1, -1, 1], [3, 2, 3])
        assert np.array_equal(c[1, 1, 0], [3, 3, 1])
        assert np.array_equal(c[1, 1, -1], [3, 3, 2])
        assert np.array_equal(c[1, 1, 1], [3, 3, 3])
        assert np.array_equal(c[2, 2, 2], [3, 3, 3])

    def test_grain_boundary_forces_one_sided_differences(self):
        """At a boundary between two grains, the completeness should switch to
        a one-sided difference that only looks into the same grain."""
        ids = np.ones((6, 4, 4), dtype=int)
        ids[3:] = 2
        c = core.get_completeness(ids)

        # Last voxel of grain 1 (x=2): only the left (backward) neighbor is
        # in the same grain.
        assert c[2, 1, 1, 0] == 2
        # First voxel of grain 2 (x=3): only the right (forward) neighbor is
        # in the same grain.
        assert c[3, 1, 1, 0] == 1
        # Interior of grain 1, away from the boundary: still central.
        assert c[1, 1, 1, 0] == 3

    def test_invalid_grain_id_zeroed(self):
        """Voxels with grain_id == 0 (unindexed) should have zero completeness
        in every direction."""
        ids = np.ones((4, 4, 4), dtype=int)
        ids[1, 1, 1] = 0
        c = core.get_completeness(ids)
        assert np.array_equal(c[1, 1, 1], [0, 0, 0])

    def test_output_shape_and_dtype(self):
        ids = np.ones((3, 5, 7), dtype=int)
        c = core.get_completeness(ids)
        assert c.shape == (3, 5, 7, 3)
        assert c.dtype == np.int8


class TestGetNeighbors:
    """Tests for get_neighbors function."""

    def test_central_difference_neighbors(self):
        """An interior voxel with completeness 3 in every direction should get
        symmetric +/-1 neighbor coordinates and a scale of 2."""
        ids = np.ones((4, 4, 4), dtype=int)
        completeness = core.get_completeness(ids)
        coords0, coords1, scale = core.get_neighbors(completeness)

        # coords0/coords1 shape: (X, Y, Z, diff_axis=3, coord=3); scale: (X, Y, Z, 3)
        assert coords0.shape == ids.shape + (3, 3)
        assert coords1.shape == ids.shape + (3, 3)
        assert scale.shape == ids.shape + (3,)

        voxel = (1, 1, 1)
        for axis in range(3):
            expected_minus = np.array(voxel)
            expected_minus[axis] -= 1
            expected_plus = np.array(voxel)
            expected_plus[axis] += 1
            assert np.array_equal(coords0[voxel][axis], expected_minus)
            assert np.array_equal(coords1[voxel][axis], expected_plus)
            assert scale[voxel][axis] == 2

    def test_one_sided_difference_neighbors(self):
        """A boundary voxel with completeness 1 (forward-only) should have no
        shift on the first point and a +1 shift on the second, with scale 1."""
        ids = np.ones((4, 4, 4), dtype=int)
        completeness = core.get_completeness(ids)
        coords0, coords1, scale = core.get_neighbors(completeness)

        voxel = (0, 1, 1)  # x=0 face: completeness[..., 0] == 1 (forward)
        assert np.array_equal(coords0[voxel][0], [0, 1, 1])
        assert np.array_equal(coords1[voxel][0], [1, 1, 1])
        assert scale[voxel][0] == 1

    def test_invalid_voxel_has_zero_scale(self):
        ids = np.ones((4, 4, 4), dtype=int)
        ids[1, 1, 1] = 0
        completeness = core.get_completeness(ids)
        _, _, scale = core.get_neighbors(completeness)
        assert np.array_equal(scale[1, 1, 1], [0, 0, 0])


class TestGetFiniteDifferenceCoordinates:
    """Tests for get_finite_difference_coordinates function."""

    def test_matches_manual_completeness_and_neighbors(self):
        """This function should just chain get_completeness and get_neighbors."""
        ids = np.ones((4, 4, 4), dtype=int)
        coords0, coords1, scale = core.get_finite_difference_coordinates(ids)

        completeness = core.get_completeness(ids)
        expected_coords0, expected_coords1, expected_scale = core.get_neighbors(completeness)

        assert np.array_equal(coords0, expected_coords0)
        assert np.array_equal(coords1, expected_coords1)
        assert np.array_equal(scale, expected_scale)


class TestMinimizeL1L2Primitives:
    """Tests for the _minimize_l2 and _minimize_l1 helper functions."""

    def test_minimize_l2_zero_input_gives_zero_output(self):
        _, B = core.get_linear_operator(cs=1, slip_systems="all")
        Lambda = np.zeros((5, 9))
        dd = core._minimize_l2(Lambda, B)
        assert np.allclose(dd, 0.0)

    def test_minimize_l2_reconstructs_alpha(self):
        A, B = core.get_linear_operator(cs=1, slip_systems="all")
        np.random.seed(30)
        rho_true = np.random.rand(A.shape[1])
        Lambda = (A @ rho_true).reshape(1, 9)
        dd = core._minimize_l2(Lambda, B).reshape(B.shape[0], -1)
        reconstructed = A @ dd
        assert np.allclose(reconstructed.flatten(), Lambda.flatten(), atol=1e-4)

    def test_minimize_l1_reconstructs_alpha(self):
        A, _ = core.get_linear_operator(cs=1, slip_systems="all")
        np.random.seed(31)
        rho_true = np.random.rand(A.shape[1])
        Lambda = (A @ rho_true).reshape(1, 9)
        dd = core._minimize_l1(Lambda, A)
        reconstructed = A @ dd
        assert np.allclose(reconstructed.flatten(), Lambda.flatten(), atol=1e-3)


class TestMinimize:
    """Tests for the top-level minimize function."""

    @staticmethod
    def _random_alpha(A, n_voxels, seed):
        """Build alpha (n_voxels, 3, 3) from a known non-negative density so
        the minimization has a legitimate reconstruction target."""
        rng = np.random.default_rng(seed)
        rho_true = rng.random((A.shape[1], n_voxels))
        lambda_flat = (A @ rho_true).T  # (n_voxels, 9)
        return lambda_flat.reshape(n_voxels, 3, 3), rho_true

    def test_minimize_l2_fcc_reconstruction(self):
        A, B = core.get_linear_operator(cs=1, slip_systems="all")
        alpha, _ = self._random_alpha(A, n_voxels=4, seed=40)
        burgers = 2.48e-10
        dd = core.minimize(alpha, cs=1, A=A, B=B, burgers=burgers, minimization="l2")
        assert dd.shape == (A.shape[1], 4)
        reconstructed = A @ (dd * burgers).reshape(A.shape[1], -1)
        assert np.allclose(reconstructed, alpha.reshape(4, 9).T, atol=1e-4)

    def test_minimize_l1_default_args_does_not_crash(self):
        """Regression test: minimize()'s default n_cpus=-1 used to be divided
        into directly when computing the default chunk_size, producing a
        negative/invalid chunk size."""
        A, B = core.get_linear_operator(cs=1, slip_systems="all")
        alpha, _ = self._random_alpha(A, n_voxels=6, seed=41)
        burgers = 2.48e-10
        dd = core.minimize(
            alpha, cs=1, A=A, B=B, burgers=burgers, minimization="l1", n_cpus=-1
        )
        assert dd.shape == (A.shape[1], 6)
        assert np.all(np.isfinite(dd))

    def test_minimize_l1_oversized_chunk_size_does_not_crash(self):
        """Regression test: an explicit chunk_size larger than the number of
        voxels used to make np.array_split's section count 0."""
        A, B = core.get_linear_operator(cs=1, slip_systems="all")
        alpha, _ = self._random_alpha(A, n_voxels=3, seed=42)
        dd = core.minimize(
            alpha,
            cs=1,
            A=A,
            B=B,
            burgers=2.48e-10,
            minimization="l1",
            n_cpus=1,
            chunk_size=1000,
        )
        assert dd.shape == (A.shape[1], 3)
        assert np.all(np.isfinite(dd))

    def test_minimize_invalid_scheme_raises(self):
        A, B = core.get_linear_operator(cs=1, slip_systems="all")
        alpha, _ = self._random_alpha(A, n_voxels=2, seed=43)
        with pytest.raises(ValueError, match="Minimization scheme not recognized"):
            core.minimize(alpha, cs=1, A=A, B=B, burgers=1.0, minimization="bogus")

    @pytest.mark.parametrize(
        "slip_systems,burgers",
        [
            ("basal", 2.48e-10),
            ("prismatic", 2.48e-10),
            ("pyramidal", 2.48e-10),
            ("basal+prismatic", 2.48e-10),
            ("basal+pyramidal", (2.48e-10, 2.5e-10)),
            ("prismatic+pyramidal", (2.48e-10, 2.5e-10)),
            ("all", (2.48e-10, 2.5e-10)),
        ],
    )
    def test_minimize_hcp_burgers_division_all_slip_combos(self, slip_systems, burgers):
        """Regression test: burgers_basal_prismatic/burgers_pyramidal used to
        only be conditionally assigned; make sure every slip-system combination
        (each hitting a different branch of that logic) runs without a crash."""
        A, B = core.get_linear_operator(cs=3, slip_systems=slip_systems)
        alpha, _ = self._random_alpha(A, n_voxels=3, seed=44)
        dd = core.minimize(alpha, cs=3, A=A, B=B, burgers=burgers, minimization="l2")
        assert dd.shape == (A.shape[1], 3)
        assert np.all(np.isfinite(dd))


class TestGetOrientationGradients:
    """Tests for get_orientation_gradients."""

    @staticmethod
    def _build_inputs(shape, euler_noise_scale, seed):
        rng = np.random.default_rng(seed)
        base_euler = np.array([0.3, 0.5, 0.2])
        euler = np.tile(base_euler, shape + (1,)).astype(np.float32)
        if euler_noise_scale:
            euler = euler + rng.normal(scale=euler_noise_scale, size=euler.shape).astype(
                np.float32
            )
        quats = rotations.eu2qu(euler)
        ids = np.ones(shape, dtype=int)
        nbrs0, nbrs1, distances = core.get_finite_difference_coordinates(ids)
        distances = distances * np.array([1e-7, 1e-7, 1e-7])
        return quats, nbrs0, nbrs1, distances

    def test_uniform_orientation_gives_zero_gradient(self):
        """A perfectly uniform orientation field has no disorientation between
        any neighboring voxels, so both the gradients and misorientations
        should be exactly zero."""
        shape = (4, 4, 4)
        quats, nbrs0, nbrs1, distances = self._build_inputs(shape, euler_noise_scale=0, seed=1)
        dphi, mis = core.get_orientation_gradients(quats, nbrs0, nbrs1, distances, cs=1, n_cpus=1)
        assert dphi.shape == shape + (3, 3)
        assert mis.shape == shape + (3,)
        assert np.allclose(mis, 0.0)
        assert np.allclose(dphi, 0.0)

    def test_single_and_multi_cpu_paths_agree_on_misorientation(self):
        """Regression test: calculate()'s default n_cpus=-1 routes through the
        parallel/chunked branch, which used to reference a deleted variable
        and crash. The misorientation *magnitude* should also agree between
        paths (up to float32 precision noise). The gradient tensor itself can
        legitimately differ in axis direction between paths, since disorientation
        picks among several crystallographically-equivalent representative
        quaternions and chunking can change which one is picked."""
        shape = (4, 4, 5)
        quats, nbrs0, nbrs1, distances = self._build_inputs(shape, euler_noise_scale=0.05, seed=50)

        dphi_serial, mis_serial = core.get_orientation_gradients(
            quats.copy(), nbrs0.copy(), nbrs1.copy(), distances.copy(), cs=1, n_cpus=1
        )
        dphi_parallel, mis_parallel = core.get_orientation_gradients(
            quats.copy(),
            nbrs0.copy(),
            nbrs1.copy(),
            distances.copy(),
            cs=1,
            n_cpus=2,
            chunk_size=None,
        )
        assert mis_serial.max() > 0  # sanity check: gradients are actually present
        assert np.allclose(mis_serial, mis_parallel, atol=1e-4)
        assert dphi_parallel.shape == dphi_serial.shape
        assert np.all(np.isfinite(dphi_parallel))

    def test_default_n_cpus_does_not_crash(self):
        """Regression test for the exact bug found in calculate(): n_cpus=-1
        (joblib's "all cores" sentinel) with chunk_size=None used to reference
        a deleted `quats` variable and crash with UnboundLocalError."""
        shape = (3, 3, 3)
        quats, nbrs0, nbrs1, distances = self._build_inputs(shape, euler_noise_scale=0.05, seed=51)
        dphi, mis = core.get_orientation_gradients(
            quats, nbrs0, nbrs1, distances, cs=1, n_cpus=-1, chunk_size=None
        )
        assert dphi.shape == shape + (3, 3)
        assert np.all(np.isfinite(dphi))
        assert np.all(np.isfinite(mis))


class TestCalculate:
    """Tests for the top-level calculate function."""

    @staticmethod
    def _synthetic_dataset(shape, seed, noise=0.05):
        rng = np.random.default_rng(seed)
        base_euler = np.array([0.3, 0.5, 0.2])
        euler = np.tile(base_euler, shape + (1,)).astype(np.float32)
        euler = euler + rng.normal(scale=noise, size=euler.shape).astype(np.float32)
        ids = np.ones(shape, dtype=int)
        return euler, ids

    def test_calculate_default_args_does_not_crash(self):
        """Regression test for the headline audit finding: calculate() crashed
        on every call with default arguments (n_cpus=-1, chunk_size=None) due
        to get_orientation_gradients referencing a deleted variable."""
        euler, ids = self._synthetic_dataset((5, 5, 5), seed=60)
        dd, mis = core.calculate(
            euler,
            ids,
            cs=1,
            slip_systems="all",
            burgers=2.48e-10,
            spacing=(1e-7, 1e-7, 1e-7),
            minimization="l2",
            progress_bar=False,
        )
        assert dd["l2"].shape == (18, 5, 5, 5)
        assert mis.shape == (3, 5, 5, 5)
        assert np.all(np.isfinite(dd["l2"]))
        assert np.all(np.isfinite(mis))

    def test_calculate_explicit_multi_cpu(self):
        euler, ids = self._synthetic_dataset((4, 4, 4), seed=61)
        dd, _ = core.calculate(
            euler,
            ids,
            cs=1,
            slip_systems="all",
            burgers=2.48e-10,
            spacing=(1e-7, 1e-7, 1e-7),
            minimization="l2",
            n_cpus=2,
            progress_bar=False,
        )
        assert dd["l2"].shape == (18, 4, 4, 4)

    def test_calculate_l1_default_args(self):
        euler, ids = self._synthetic_dataset((3, 3, 3), seed=62)
        dd, _ = core.calculate(
            euler,
            ids,
            cs=1,
            slip_systems="all",
            burgers=2.48e-10,
            spacing=(1e-7, 1e-7, 1e-7),
            minimization="l1",
            progress_bar=False,
        )
        assert dd["l1"].shape == (18, 3, 3, 3)
        assert np.all(np.isfinite(dd["l1"]))

    def test_calculate_both_minimizations(self):
        euler, ids = self._synthetic_dataset((3, 3, 3), seed=63)
        dd, _ = core.calculate(
            euler,
            ids,
            cs=1,
            slip_systems="all",
            burgers=2.48e-10,
            spacing=(1e-7, 1e-7, 1e-7),
            minimization=["l1", "l2"],
            progress_bar=False,
        )
        assert set(dd.keys()) == {"l1", "l2"}

    def test_calculate_invalid_euler_shape_raises(self):
        euler = np.zeros((3, 3, 3, 4))
        ids = np.ones((3, 3, 3), dtype=int)
        with pytest.raises(ValueError, match="Euler angles must have shape"):
            core.calculate(euler, ids, cs=1, slip_systems="all", burgers=1.0, spacing=(1, 1, 1))

    def test_calculate_mismatched_ids_shape_raises(self):
        euler = np.zeros((3, 3, 3, 3))
        ids = np.ones((3, 3), dtype=int)
        with pytest.raises(ValueError, match="grain IDs must have the same shape"):
            core.calculate(euler, ids, cs=1, slip_systems="all", burgers=1.0, spacing=(1, 1, 1))

    def test_calculate_invalid_cs_raises(self):
        euler = np.zeros((3, 3, 3, 3))
        ids = np.ones((3, 3, 3), dtype=int)
        with pytest.raises(ValueError, match="crystal structure must be"):
            core.calculate(euler, ids, cs=5, slip_systems="all", burgers=1.0, spacing=(1, 1, 1))

    def test_calculate_invalid_spacing_length_raises(self):
        euler = np.zeros((3, 3, 3, 3))
        ids = np.ones((3, 3, 3), dtype=int)
        with pytest.raises(ValueError, match="spacing must have"):
            core.calculate(euler, ids, cs=1, slip_systems="all", burgers=1.0, spacing=(1, 1))


def create_ang_file(tmp_path, nrows=2, ncols=3, noise_seed=None):
    """Create a minimal valid .ang file for calculate_and_save tests. If
    noise_seed is given, small per-point Euler angle noise is added so the
    resulting GND density is nonzero; otherwise every point is identical."""
    lines = [
        "# XSTEP: 0.500000",
        "# YSTEP: 0.500000",
        f"# NCOLS_ODD: {ncols}",
        f"# NCOLS_EVEN: {ncols}",
        f"# NROWS: {nrows}",
        "# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase index",
    ]
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    path = tmp_path / "test.ang"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for r in range(nrows):
            for c in range(ncols):
                euler = [0.3, 0.5, 0.2]
                if rng is not None:
                    euler = list(np.array(euler) + rng.normal(scale=0.05, size=3))
                row = euler + [float(c), float(r), 100.0, 0.5, 1]
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    return path


class TestCalculateAndSave:
    """Tests for the top-level calculate_and_save function."""

    def test_ang_path_end_to_end(self, tmp_path):
        """Runs the full ang -> calculate -> save_npz/generate_images path with
        default n_cpus=-1, chunk_size=1000 (both regression-relevant defaults),
        with enough orientation variation to give a nonzero GND density."""
        ang_path = create_ang_file(tmp_path, nrows=3, ncols=3, noise_seed=70)
        result = core.calculate_and_save(
            cs=1,
            burgers=2.48e-10,
            ang_path=ang_path,
            slip_systems="all",
            minimization="l2",
            progress_bar=False,
        )
        assert result is True
        assert (tmp_path / "gnd_l2.npy").exists()
        assert (tmp_path / "fdm.npy").exists()
        assert (tmp_path / "images").exists()
        gnd = np.load(tmp_path / "gnd_l2.npy")
        assert np.any(gnd > 0)

    def test_ang_path_all_zero_gnd_does_not_crash(self, tmp_path):
        """Regression test: a perfectly uniform orientation field gives an
        all-zero GND density, and the results-summary printing used to crash
        with `dd[m][dd[m] > 0].min()` on the resulting empty array."""
        ang_path = create_ang_file(tmp_path, nrows=3, ncols=3)
        result = core.calculate_and_save(
            cs=1,
            burgers=2.48e-10,
            ang_path=ang_path,
            slip_systems="all",
            minimization="l2",
            progress_bar=False,
        )
        assert result is True

    def test_neither_path_provided_raises(self):
        """Regression test: previously euler/ids/spacing were never assigned if
        neither dream3d_path nor ang_path was given, raising an unhelpful
        UnboundLocalError instead of a clear ValueError."""
        with pytest.raises(ValueError, match="Either dream3d_path or ang_path"):
            core.calculate_and_save(cs=1, burgers=2.48e-10)

    def test_dream3d_path_without_names_raises(self):
        with pytest.raises(ValueError, match="ids_name and euler_name"):
            core.calculate_and_save(cs=1, burgers=2.48e-10, dream3d_path="some.dream3d")
