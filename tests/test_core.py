"""Tests for the core module."""

import numpy as np
import pytest
from pygnd import core


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
