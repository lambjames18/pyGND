"""Tests for the IO module."""

import h5py
import numpy as np
import pytest
from pygnd import io


def create_dream3d_dummy_file(tmp_path):
    """Create a dummy DREAM.3D file for testing."""
    file_path = tmp_path / "dummy.dream3d"
    # Layered structure of a DREAM.3D file
    # File
    #  \__ /DataContainers
    #      \__ /ImageDataContainer
    #            \__ /CellData
    #                 \__ /FeatureIds (dataset)
    #                 \__ /EulerAngles (dataset)
    #  \__ /_SIMPL_GEOMETRY
    #       \__ /SPACING (dataset)

    shape = (5, 5, 5)
    with h5py.File(file_path, "w") as f:
        data_container = f.create_group("DataContainers/ImageDataContainer/CellData")
        feature_ids = np.arange(np.prod(shape), dtype=np.int32).reshape(shape + (1,))
        euler_angles = np.random.rand(np.prod(shape), 3).astype(np.float32).reshape(shape + (3,))
        data_container.create_dataset("FeatureIds", data=feature_ids)
        data_container.create_dataset("EulerAngles", data=euler_angles)

        simpl_geometry = f.create_group("_SIMPL_GEOMETRY")
        spacing = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        simpl_geometry.create_dataset("SPACING", data=spacing)
        # Add dummy attribute to _SIMPL_GEOMETRY
        simpl_geometry.attrs["TEST"] = np.array([1], dtype=np.float32)
    return file_path


class TestReadDREAM3DData:
    """Tests for read_dream3d function."""

    def test_read_valid_file(self, tmp_path):
        """Test reading a valid DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        ids_name = "FeatureIds"
        euler_name = "EulerAngles"
        spacing_units = "um"
        euler, ids, spacing = io.read_dream3d(
            dream3d_file, ids_name, euler_name, spacing_units
        )  # Should not raise an exception
        assert euler.ndim == 4
        assert ids.ndim == 3
        assert len(spacing) == 3

    def test_read_invalid_file(self):
        """Test reading an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.read_dream3d("non_existent_file.dream3d", "FeatureIds", "EulerAngles")

    def test_invalid_ids_name(self, tmp_path):
        """Test reading with an invalid Feature IDs name."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        with pytest.raises(KeyError):
            io.read_dream3d(dream3d_file, "InvalidIDs", "EulerAngles")

    def test_invalid_euler_name(self, tmp_path):
        """Test reading with an invalid Euler Angles name."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        with pytest.raises(KeyError):
            io.read_dream3d(dream3d_file, "FeatureIds", "InvalidEuler")


class TestSaveDREAM3DData:
    """Tests for save_dream3d function."""

    def test_save_and_read_back(self, tmp_path):
        """Test saving data to a DREAM.3D file and reading it back."""
        # Create dummy h5 file
        shape = (2, 2, 2)
        h5_file_path = tmp_path / "test.dream3d"
        with h5py.File(h5_file_path, "w") as f:
            f.create_dataset(
                "/DataContainer/CellData/FeatureIds",
                data=np.arange(8, dtype=np.float32).reshape(shape + (1,)),
            )

        # Data to save
        gnd = {"l1": np.random.rand(18, *shape).astype(np.float32)}
        fdm = np.random.rand(3, *shape).astype(np.float32)

        # Create expected output
        l1_data_expected = gnd["l1"].sum(axis=0).reshape(shape + (1,))
        fdm_avg_expected = fdm.mean(axis=0).reshape(shape + (1,))
        fdm_max_expected = fdm.max(axis=0).reshape(shape + (1,))

        # Save data
        io.save_to_dream3d(h5_file_path, "FeatureIds", gnd, fdm)

        # Read back data
        with h5py.File(h5_file_path, "r") as f:
            l1_data_returned = f["/DataContainer/CellData/GND_l1"][:]
            fdm_avg_returned = f["/DataContainer/CellData/FDM_avg"][:]
            fdm_max_returned = f["/DataContainer/CellData/FDM_max"][:]

        assert np.array_equal(l1_data_returned, l1_data_expected)
        assert np.array_equal(fdm_avg_returned, fdm_avg_expected)
        assert np.array_equal(fdm_max_returned, fdm_max_expected)

        # Clean up
        h5_file_path.unlink()

    def test_save_invalid_file(self, tmp_path):
        """Test saving data to an invalid DREAM.3D file."""
        invalid_path = tmp_path / "non_existent_dir" / "test.dream3d"
        gnd = {"l1": np.random.rand(18, 2, 2, 2).astype(np.float32)}
        fdm = np.random.rand(3, 2, 2, 2).astype(np.float32)
        with pytest.raises(FileNotFoundError):
            io.save_to_dream3d(invalid_path, "FeatureIds", gnd, fdm)

    def test_save_invalid_ids_name(self, tmp_path):
        """Test saving data with an invalid Feature IDs name."""
        # Create dummy h5 file
        shape = (2, 2, 2)
        h5_file_path = tmp_path / "test.dream3d"
        with h5py.File(h5_file_path, "w") as f:
            f.create_dataset(
                "/DataContainer/CellData/FeatureIds",
                data=np.arange(8, dtype=np.float32).reshape(shape + (1,)),
            )

        gnd = {"l1": np.random.rand(18, *shape).astype(np.float32)}
        fdm = np.random.rand(3, *shape).astype(np.float32)

        with pytest.raises(KeyError):
            io.save_to_dream3d(h5_file_path, "InvalidIDs", gnd, fdm)

        h5_file_path.unlink()


class TestReadDREAM3DSpacing:
    """Tests for read_dream3d_spacing function."""

    def test_read_valid_spacing(self, tmp_path):
        """Test reading spacing from a valid DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        spacing_units = "um"
        spacing = io.read_dream3d_spacing(dream3d_file, spacing_units)
        assert len(spacing) == 3
        assert all(s == 1.5e-6 for s in spacing)  # Assuming known spacing values

    def test_no_spacing_in_file(self, tmp_path):
        """Test reading spacing from a DREAM.3D file with no spacing info."""
        # Create dummy h5 file without spacing
        h5_file_path = tmp_path / "test_no_spacing.dream3d"
        with h5py.File(h5_file_path, "w") as f:
            f.create_dataset(
                "/DataContainer/CellData/FeatureIds",
                data=np.arange(8, dtype=np.float32).reshape((2, 2, 2, 1)),
            )

        with pytest.raises(ValueError):
            io.read_dream3d_spacing(h5_file_path, "um")

        h5_file_path.unlink()

    def test_invalid_file(self):
        """Test reading spacing from an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.read_dream3d_spacing("non_existent_file.dream3d")

    def test_invalid_spacing_units(self, tmp_path):
        """Test reading spacing with invalid units."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        with pytest.raises(ValueError):
            io.read_dream3d_spacing(dream3d_file, "invalid_unit")

    def test_spacing_units(self, tmp_path):
        """Test reading spacing with different valid units."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        units_and_expected = {
            "m": 1.5,
            "meter": 1.5,
            "meters": 1.5,
            "mm": 1.5e-3,
            "millimeter": 1.5e-3,
            "millimeters": 1.5e-3,
            "um": 1.5e-6,
            "micron": 1.5e-6,
            "microns": 1.5e-6,
            "micrometer": 1.5e-6,
            "micrometers": 1.5e-6,
            "µm": 1.5e-6,
            "nm": 1.5e-9,
            "nanometer": 1.5e-9,
            "nanometers": 1.5e-9,
        }
        for units, expected in units_and_expected.items():
            spacing = io.read_dream3d_spacing(dream3d_file, units)
            assert all(np.isclose(s, expected) for s in spacing)


class TestExtractDataFromH5:
    """Tests for extract_data_from_h5 function."""

    def test_extract_valid_data(self, tmp_path):
        """Test extracting valid data from a DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        data = io.extract_data_from_h5(dream3d_file, "FeatureIds")
        assert data is not None
        assert data.ndim == 4

    def test_extract_invalid_data(self, tmp_path):
        """Test extracting invalid data from a DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        data = io.extract_data_from_h5(dream3d_file, "InvalidDataName")
        assert data is None

    def test_invalid_file(self):
        """Test extracting data from an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.extract_data_from_h5("non_existent_file.dream3d", "FeatureIds")


class TestExtractAttributeFromH5:
    """Tests for extract_attribute_from_h5 function."""

    def test_extract_valid_attribute(self, tmp_path):
        """Test extracting a valid attribute from a DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        attribute = io.extract_attribute_from_h5(dream3d_file, "TEST")
        assert attribute is not None
        assert isinstance(attribute, np.ndarray)

    def test_extract_invalid_attribute(self, tmp_path):
        """Test extracting an invalid attribute from a DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        attribute = io.extract_attribute_from_h5(dream3d_file, "InvalidAttributeName")
        assert attribute is None

    def test_invalid_file(self):
        """Test extracting an attribute from an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.extract_attribute_from_h5("non_existent_file.dream3d", "TupleDimensions")


class TestExtractPathFromH5:
    """Tests for extract_path_from_h5 function."""

    def test_extract_valid_path(self, tmp_path):
        """Test extracting a valid path from a DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        path = io.extract_path_from_h5(dream3d_file, "EulerAngles")
        assert path is not None
        assert isinstance(path, str)

    def test_extract_invalid_path(self, tmp_path):
        """Test extracting an invalid path from a DREAM.3D file."""
        dream3d_file = create_dream3d_dummy_file(tmp_path)
        path = io.extract_path_from_h5(dream3d_file, "Invalid/Path/Name")
        assert path is None

    def test_invalid_file(self):
        """Test extracting a path from an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.extract_path_from_h5("non_existent_file.dream3d", "EulerAngles")


class TestAddDataToH5:
    """Tests for add_dataset_to_h5 function."""

    def test_add_dataset(self, tmp_path):
        """Test adding data to an HDF5 file."""
        h5_file_path = tmp_path / "test.h5"
        data_to_add = np.random.rand(4, 4, 4, 1).astype(np.float32)
        dataset_name = "TestData"

        h5_group = h5py.File(h5_file_path, "w")

        # Add data
        io.add_dataset_to_h5(h5_group, dataset_name, data_to_add)
        h5_group.close()

        # Verify data was added
        with h5py.File(h5_file_path, "r") as f:
            assert dataset_name in f
            np.testing.assert_array_equal(f[dataset_name][:], data_to_add)

        h5_file_path.unlink()

    def test_overwrite_dataset(self, tmp_path):
        """Test adding data to an HDF5 file."""
        h5_file_path = tmp_path / "test.h5"
        data_to_add_original = np.random.rand(4, 4, 4, 1).astype(np.float32)
        dataset_name = "TestData"

        h5_group = h5py.File(h5_file_path, "w")

        # Add data
        io.add_dataset_to_h5(h5_group, dataset_name, data_to_add_original)
        h5_group.close()

        # Overwrite data
        data_to_add = np.random.rand(4, 4, 4, 1).astype(np.float32)
        h5_group = h5py.File(h5_file_path, "r+")
        io.add_dataset_to_h5(h5_group, dataset_name, data_to_add)
        h5_group.close()

        # Verify data was added
        with h5py.File(h5_file_path, "r") as f:
            assert dataset_name in f
            np.testing.assert_array_equal(f[dataset_name][:], data_to_add)

        h5_file_path.unlink()

    def test_add_dataset_invalid_file(self):
        """Test adding data to an invalid HDF5 file path."""
        not_a_h5_group = "not_a_h5_group"
        data_to_add = np.random.rand(4, 4, 4, 1).astype(np.float32)
        dataset_name = "TestData"

        with pytest.raises(TypeError):
            io.add_dataset_to_h5(not_a_h5_group, dataset_name, data_to_add)

    def test_add_dataset_invalid_data(self, tmp_path):
        """Test adding invalid data to an HDF5 file."""
        h5_file_path = tmp_path / "test.h5"
        data_to_add = np.random.rand(4, 4, 4, 1).astype(np.float32)
        dataset_name = "TestData"

        h5_group = h5py.File(h5_file_path, "w")
        io.add_dataset_to_h5(h5_group, dataset_name, data_to_add)

        bad_data_to_add = np.random.rand(4, 4, 5, 1).astype(np.float32)
        with pytest.raises(ValueError):
            io.add_dataset_to_h5(h5_group, dataset_name, bad_data_to_add)
        h5_group.close()

        h5_file_path.unlink()

    def test_add_dataset_invalid_dtype(self, tmp_path):
        """Test adding data with unsupported dtype to an HDF5 file."""
        h5_file_path = tmp_path / "test.h5"
        data_to_add = np.random.rand(4, 4, 4, 1).astype(np.complex64)  # Unsupported dtype
        dataset_name = "TestData"

        h5_group = h5py.File(h5_file_path, "w")

        with pytest.raises(TypeError):
            io.add_dataset_to_h5(h5_group, dataset_name, data_to_add)
        h5_group.close()

        h5_file_path.unlink()


def create_ang_file(tmp_path, nrows=2, ncols=3, extra_columns=0, include_column_headers=True):
    """Create a synthetic .ang file for testing read_ang."""
    n_entries = 8 + extra_columns
    lines = []
    lines.append(f"# XSTEP: 0.500000")
    lines.append(f"# YSTEP: 0.500000")
    lines.append(f"# NCOLS_ODD: {ncols}")
    lines.append(f"# NCOLS_EVEN: {ncols}")
    lines.append(f"# NROWS: {nrows}")
    if include_column_headers:
        base_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]
        names = base_names + [f"extra_{i}" for i in range(extra_columns)]
        lines.append("# COLUMN_HEADERS: " + ", ".join(names))

    path = tmp_path / "test.ang"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for r in range(nrows):
            for c in range(ncols):
                row = [0.1, 0.2, 0.3, float(c), float(r), 100.0, 0.5, 1] + [0.0] * extra_columns
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    return path


class TestReadAng:
    """Tests for read_ang function."""

    def test_read_valid_file(self, tmp_path):
        path = create_ang_file(tmp_path, nrows=2, ncols=3)
        euler, ids, spacing = io.read_ang(path)
        assert euler.shape == (1, 2, 3, 3)
        assert ids.shape == (1, 2, 3)
        assert np.allclose(ids, 1)  # no ids_path given -> all ones
        assert np.allclose(spacing, 0.5e-6)

    def test_euler_angle_values(self, tmp_path):
        path = create_ang_file(tmp_path, nrows=2, ncols=3)
        euler, _, _ = io.read_ang(path)
        assert np.allclose(euler[0, ..., 0], 0.1)
        assert np.allclose(euler[0, ..., 1], 0.2)
        assert np.allclose(euler[0, ..., 2], 0.3)

    def test_extra_columns_without_column_headers(self, tmp_path):
        """A file with more than 8 data columns and no explicit COLUMN_HEADERS
        line should not raise IndexError (regression test)."""
        path = create_ang_file(
            tmp_path, nrows=2, ncols=3, extra_columns=2, include_column_headers=False
        )
        euler, ids, spacing = io.read_ang(path)
        assert euler.shape == (1, 2, 3, 3)

    def test_missing_header_field_raises(self, tmp_path):
        """A header missing NCOLS_ODD/NROWS/XSTEP should raise a clear
        ValueError instead of UnboundLocalError (regression test)."""
        path = tmp_path / "bad.ang"
        with open(path, "w") as f:
            f.write("# YSTEP: 0.500000\n# NROWS: 2\n")
            for _ in range(6):
                f.write(" ".join(["0.0"] * 8) + "\n")
        with pytest.raises(ValueError, match="missing"):
            io.read_ang(path)

    def test_grid_size_mismatch_raises(self, tmp_path):
        path = create_ang_file(tmp_path, nrows=2, ncols=3)
        # Corrupt the file by appending an extra data row that breaks the grid size
        with open(path, "a") as f:
            f.write(" ".join(["0.0"] * 8) + "\n")
        with pytest.raises(ValueError, match="does not match the expected grid"):
            io.read_ang(path)

    def test_with_ids_path(self, tmp_path):
        path = create_ang_file(tmp_path, nrows=2, ncols=3)
        ids_path = tmp_path / "grain_ids.txt"
        # 6 points (2x3), column index 8 holds the grain id
        with open(ids_path, "w") as f:
            for i in range(6):
                row = [0.0] * 8 + [float(i % 2)]
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        euler, ids, spacing = io.read_ang(path, ids_path=ids_path)
        assert ids.shape == (1, 2, 3)
        assert set(np.unique(ids).tolist()) == {0, 1}


class TestNpzSaveRemove:
    """Tests for save_npz and remove_npz."""

    def test_save_and_remove(self, tmp_path):
        gnd_data = {"l1": np.random.rand(18, 4, 4), "l2": np.random.rand(18, 4, 4)}
        fdm_data = np.random.rand(3, 4, 4)
        io.save_npz(gnd_data, fdm_data, tmp_path)

        assert (tmp_path / "gnd_l1.npy").exists()
        assert (tmp_path / "gnd_l2.npy").exists()
        assert (tmp_path / "fdm.npy").exists()

        saved_l1 = np.load(tmp_path / "gnd_l1.npy")
        assert np.allclose(saved_l1, gnd_data["l1"])

        io.remove_npz(tmp_path)
        assert not (tmp_path / "gnd_l1.npy").exists()
        assert not (tmp_path / "gnd_l2.npy").exists()
        assert not (tmp_path / "fdm.npy").exists()

    def test_remove_npz_missing_files_does_not_raise(self, tmp_path):
        """remove_npz on a folder with no .npy files should be a no-op."""
        io.remove_npz(tmp_path)  # should not raise


class TestGenerateImages:
    """Tests for generate_images."""

    def test_generates_expected_png_files(self, tmp_path):
        gnd_data = {"l2": np.random.rand(3, 8, 8) + 1e-6}
        fdm_data = np.random.rand(3, 8, 8) + 1e-6
        io.generate_images(gnd_data=gnd_data, fdm_data=fdm_data, folder=tmp_path)

        images_dir = tmp_path / "images"
        assert images_dir.exists()
        for i in range(3):
            assert (images_dir / f"gnd_l2_{i}.png").exists()
        assert (images_dir / "gnd_l2_sum.png").exists()
        assert (images_dir / "fdm_avg.png").exists()
        assert (images_dir / "fdm_max.png").exists()


def create_test_xdmf(tmp_path, dims=(2, 2, 2), filename="test.dream3d.xdmf"):
    """Create a minimal synthetic XDMF file matching the demo data's structure."""
    topo_dims = " ".join(str(d + 1) for d in dims) + " "
    content = f"""<?xml version="1.0"?>
<Xdmf>
 <Domain>
  <Grid Name="ImageDataContainer" GridType="Uniform">
    <Topology TopologyType="3DCoRectMesh" Dimensions="{topo_dims}"></Topology>
    <Attribute Name="FeatureIds" AttributeType="Scalar" Center="Cell">
      <DataItem Format="HDF" Dimensions="{dims[0]} {dims[1]} {dims[2]} 1" NumberType="Int" Precision="4" >
        test.dream3d:/DataContainers/ImageDataContainer/CellData/FeatureIds
      </DataItem>
    </Attribute>
  </Grid>
 </Domain>
</Xdmf>
"""
    path = tmp_path / filename
    path.write_text(content)
    return path


class TestAddDatasetToXdmf:
    """Tests for add_dataset_to_xdmf."""

    def test_add_new_scalar_dataset(self, tmp_path):
        xdmf_path = create_test_xdmf(tmp_path, dims=(2, 2, 2))
        data = np.random.rand(2, 2, 2).astype(np.float32)
        io.add_dataset_to_xdmf(xdmf_path, "NewData", data)
        content = xdmf_path.read_text()
        assert 'Attribute Name="NewData"' in content
        assert 'AttributeType="Scalar"' in content.split('Name="NewData"')[1].split(">")[0]

    def test_skip_existing_dataset(self, tmp_path):
        xdmf_path = create_test_xdmf(tmp_path, dims=(2, 2, 2))
        data = np.random.rand(2, 2, 2).astype(np.float32)
        io.add_dataset_to_xdmf(xdmf_path, "FeatureIds", data)  # already exists
        content = xdmf_path.read_text()
        assert content.count('Attribute Name="FeatureIds"') == 1

    def test_dimension_mismatch_raises(self, tmp_path):
        xdmf_path = create_test_xdmf(tmp_path, dims=(2, 2, 2))
        bad_data = np.random.rand(3, 3, 3).astype(np.float32)
        with pytest.raises(ValueError, match="not compatible"):
            io.add_dataset_to_xdmf(xdmf_path, "BadData", bad_data)

    def test_too_few_dims_raises(self, tmp_path):
        xdmf_path = create_test_xdmf(tmp_path, dims=(2, 2, 2))
        bad_data = np.random.rand(2, 2).astype(np.float32)
        with pytest.raises(ValueError, match="at least 3-dimensional"):
            io.add_dataset_to_xdmf(xdmf_path, "BadData", bad_data)

    def test_too_many_dims_raises(self, tmp_path):
        xdmf_path = create_test_xdmf(tmp_path, dims=(2, 2, 2))
        bad_data = np.random.rand(2, 2, 2, 3, 1).astype(np.float32)
        with pytest.raises(ValueError, match="at most 4-dimensional"):
            io.add_dataset_to_xdmf(xdmf_path, "BadData", bad_data)


class TestExtractPathFromH5DatasetCollision:
    """Regression test for the endswith substring-collision bug: a DREAM3D file
    commonly has both a per-voxel array (e.g. EulerAngles) and a per-grain
    average array (e.g. AvgEulerAngles); lookups must not conflate the two."""

    def test_does_not_match_avg_prefixed_dataset(self, tmp_path):
        h5_file_path = tmp_path / "collision.dream3d"
        with h5py.File(h5_file_path, "w") as f:
            # Name this group to sort alphabetically before the real one, so
            # traversal visits the colliding dataset first.
            avg_group = f.create_group("AAA_CellFeatureData")
            avg_group.create_dataset(
                "AvgEulerAngles", data=np.zeros((5, 3), dtype=np.float32) + 99.0
            )
            cell_group = f.create_group("ZZZ_CellData")
            cell_group.create_dataset(
                "EulerAngles", data=np.ones((2, 2, 2, 3), dtype=np.float32)
            )

        path = io.extract_path_from_h5(h5_file_path, "EulerAngles")
        assert path == "/ZZZ_CellData/EulerAngles"

        data = io.extract_data_from_h5(h5_file_path, "EulerAngles")
        assert data.shape == (2, 2, 2, 3)
        assert np.allclose(data, 1.0)
