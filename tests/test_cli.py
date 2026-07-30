"""Tests for the pygnd_calculate command line interface."""

import argparse

import h5py
import numpy as np
import pytest

from pygnd import cli


def create_ang_file(tmp_path, nrows=2, ncols=3, noise_seed=None):
    """Create a minimal valid .ang file. If noise_seed is given, small
    per-point Euler angle noise is added so the resulting GND density is
    nonzero; otherwise every point is identical."""
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


def create_dream3d_file(tmp_path, shape=(1, 3, 3), noise_seed=None):
    """Create a minimal valid DREAM3D file. If noise_seed is given, small
    per-voxel Euler angle noise is added so the resulting GND density is
    nonzero; otherwise every voxel is identical."""
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    base_euler = np.array([0.3, 0.5, 0.2])
    euler = np.tile(base_euler, shape + (1,)).astype(np.float32)
    if rng is not None:
        euler = euler + rng.normal(scale=0.05, size=euler.shape).astype(np.float32)
    feature_ids = np.ones(shape + (1,), dtype=np.int32)

    path = tmp_path / "test.dream3d"
    with h5py.File(path, "w") as f:
        cell_data = f.create_group("DataContainers/ImageDataContainer/CellData")
        cell_data.create_dataset("FeatureIds", data=feature_ids)
        cell_data.create_dataset("EulerAngles", data=euler)
        geometry = f.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("SPACING", data=np.array([0.5, 0.5, 0.5], dtype=np.float32))
    return path


class TestParseBurgers:
    """Tests for the _parse_burgers helper."""

    def test_single_value(self):
        assert cli._parse_burgers("2.48e-10") == 2.48e-10

    def test_two_values(self):
        assert cli._parse_burgers("2.48e-10,2.5e-10") == (2.48e-10, 2.5e-10)

    def test_too_many_values_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._parse_burgers("1,2,3")

    def test_non_numeric_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._parse_burgers("not-a-number")


class TestMainAng:
    """Tests for `pygnd_calculate ang`."""

    def test_end_to_end(self, tmp_path):
        ang_path = create_ang_file(tmp_path, nrows=3, ncols=3, noise_seed=80)
        exit_code = cli.main(
            ["ang", str(ang_path), "--cs", "1", "--burgers", "2.48e-10", "--minimization", "l2"]
        )
        assert exit_code == 0
        assert (tmp_path / "gnd_l2.npy").exists()

    def test_mixed_minimization(self, tmp_path):
        ang_path = create_ang_file(tmp_path, nrows=2, ncols=2, noise_seed=81)
        exit_code = cli.main(
            [
                "ang",
                str(ang_path),
                "--cs",
                "1",
                "--burgers",
                "2.48e-10",
                "--minimization",
                "l1",
                "l2",
            ]
        )
        assert exit_code == 0
        assert (tmp_path / "gnd_l1.npy").exists()
        assert (tmp_path / "gnd_l2.npy").exists()

    def test_hcp_mixed_burgers(self, tmp_path):
        """Exercises the two-value --burgers form end to end with a mixed HCP
        slip-system combination."""
        ang_path = create_ang_file(tmp_path, nrows=2, ncols=2, noise_seed=82)
        exit_code = cli.main(
            [
                "ang",
                str(ang_path),
                "--cs",
                "3",
                "--burgers",
                "2.48e-10,2.5e-10",
                "--slip-systems",
                "basal+pyramidal",
            ]
        )
        assert exit_code == 0

    def test_nonexistent_file_returns_clean_error(self, tmp_path, capsys):
        exit_code = cli.main(
            ["ang", str(tmp_path / "missing.ang"), "--cs", "1", "--burgers", "2.48e-10"]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


class TestMainDream3d:
    """Tests for `pygnd_calculate dream3d`."""

    def test_end_to_end(self, tmp_path):
        dream3d_path = create_dream3d_file(tmp_path, noise_seed=83)
        exit_code = cli.main(
            [
                "dream3d",
                str(dream3d_path),
                "--ids-name",
                "FeatureIds",
                "--euler-name",
                "EulerAngles",
                "--cs",
                "1",
                "--burgers",
                "2.48e-10",
            ]
        )
        assert exit_code == 0
        with h5py.File(dream3d_path, "r") as f:
            assert "DataContainers/ImageDataContainer/CellData/GND_l2" in f

    def test_invalid_ids_name_returns_clean_error(self, tmp_path, capsys):
        dream3d_path = create_dream3d_file(tmp_path)
        exit_code = cli.main(
            [
                "dream3d",
                str(dream3d_path),
                "--ids-name",
                "NotARealName",
                "--euler-name",
                "EulerAngles",
                "--cs",
                "1",
                "--burgers",
                "1.0",
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_nonexistent_file_returns_clean_error(self, capsys):
        exit_code = cli.main(
            [
                "dream3d",
                "/nonexistent.dream3d",
                "--ids-name",
                "X",
                "--euler-name",
                "Y",
                "--cs",
                "1",
                "--burgers",
                "1.0",
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


class TestArgparseSurface:
    """Tests for the argument parser itself."""

    def test_version_flag_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "pygnd" in captured.out

    def test_missing_required_cs_exits_nonzero(self, tmp_path):
        ang_path = create_ang_file(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["ang", str(ang_path), "--burgers", "1.0"])
        assert exc_info.value.code != 0

    def test_no_command_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([])
        assert exc_info.value.code != 0

    def test_invalid_cs_choice_exits_nonzero(self, tmp_path):
        ang_path = create_ang_file(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["ang", str(ang_path), "--cs", "5", "--burgers", "1.0"])
        assert exc_info.value.code != 0
