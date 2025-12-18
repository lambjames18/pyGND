"""Tests for the IO module."""

import h5py
import numpy as np
import pytest
from pygnd import io


class TestReadDREAM3DData:
    """Tests for read_dream3d function."""

    def test_read_valid_file(self, dream3d_file):
        """Test reading a valid DREAM.3D file."""
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

    def test_invalid_ids_name(self, dream3d_file):
        """Test reading with an invalid Feature IDs name."""
        with pytest.raises(KeyError):
            io.read_dream3d(dream3d_file, "InvalidIDs", "EulerAngles")

    def test_invalid_euler_name(self, dream3d_file):
        """Test reading with an invalid Euler Angles name."""
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

    def test_read_valid_spacing(self, dream3d_file):
        """Test reading spacing from a valid DREAM.3D file."""
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

    def test_invalid_spacing_units(self, dream3d_file):
        """Test reading spacing with invalid units."""
        with pytest.raises(ValueError):
            io.read_dream3d_spacing(dream3d_file, "invalid_unit")

    def test_spacing_units(self, dream3d_file):
        """Test reading spacing with different valid units."""
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

    def test_extract_valid_data(self, dream3d_file):
        """Test extracting valid data from a DREAM.3D file."""
        data = io.extract_data_from_h5(dream3d_file, "FeatureIds")
        assert data is not None
        assert data.ndim == 4

    def test_extract_invalid_data(self, dream3d_file):
        """Test extracting invalid data from a DREAM.3D file."""
        data = io.extract_data_from_h5(dream3d_file, "InvalidDataName")
        assert data is None

    def test_invalid_file(self):
        """Test extracting data from an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.extract_data_from_h5("non_existent_file.dream3d", "FeatureIds")


class TestExtractAttributeFromH5:
    """Tests for extract_attribute_from_h5 function."""

    def test_extract_valid_attribute(self, dream3d_file):
        """Test extracting a valid attribute from a DREAM.3D file."""
        attribute = io.extract_attribute_from_h5(dream3d_file, "Pipeline Version")
        assert attribute is not None
        assert isinstance(attribute, np.ndarray)

    def test_extract_invalid_attribute(self, dream3d_file):
        """Test extracting an invalid attribute from a DREAM.3D file."""
        attribute = io.extract_attribute_from_h5(dream3d_file, "InvalidAttributeName")
        assert attribute is None

    def test_invalid_file(self):
        """Test extracting an attribute from an invalid DREAM.3D file."""
        with pytest.raises(FileNotFoundError):
            io.extract_attribute_from_h5("non_existent_file.dream3d", "TupleDimensions")


class TestExtractPathFromH5:
    """Tests for extract_path_from_h5 function."""

    def test_extract_valid_path(self, dream3d_file):
        """Test extracting a valid path from a DREAM.3D file."""
        path = io.extract_path_from_h5(dream3d_file, "EulerAngles")
        assert path is not None
        assert isinstance(path, str)

    def test_extract_invalid_path(self, dream3d_file):
        """Test extracting an invalid path from a DREAM.3D file."""
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

    def test_add_dataset_invalid_file(self, tmp_path):
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
