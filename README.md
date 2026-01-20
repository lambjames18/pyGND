# PyGND

**Geometrically Necessary Dislocation calculations for TriBeam 3D microstructures**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/PollockGroup/TriBeam_GND/actions/workflows/tests.yml/badge.svg)](https://github.com/PollockGroup/TriBeam_GND/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/PollockGroup/TriBeam_GND/branch/main/graph/badge.svg)](https://codecov.io/gh/PollockGroup/TriBeam_GND)

PyGND is a Python package for calculating geometrically necessary dislocation (GND) densities from EBSD (Electron Backscatter Diffraction) data using Nye's dislocation theory. The code was originally developed in MATLAB by Wyatt Witzen and has been reimplemented in Python for improved performance and accessibility.

## Features

- **Multiple crystal structures**: FCC, BCC, and HCP
- **Flexible slip system selection**: Choose specific slip systems or use all available
- **Optimized calculations**: Parallel processing with configurable CPU cores
- **Multiple minimization methods**: L1 and L2 minimization
- **Multiple input formats**: Support for ANG files and DREAM.3D formats
- **Progress tracking**: Built-in progress bars for long calculations
- **Memory efficient**: Configurable chunking for large datasets

## Installation

### From PyPI (once published)

```bash
pip install pygnd
```

### From source

```bash
git clone https://github.com/yourusername/TriBeam_GND.git
cd TriBeam_GND
pip install -e .
```

## Quick Start

### Example calculation for an ANG file

```python
import pygnd

# Calculate GND from an ANG file
pygnd.calculate_and_save(
    ang_path="path/to/data.ang",
    grain_ids_path="path/to/grain_data.txt",
    cs=1,  # 1=FCC, 2=BCC, 3=HCP
    burgers=2.48e-10,  # Burgers vector magnitude in meters
    minimization=["l2", "l1"],
    slip_systems="all",
    n_cpus=4,
    progress_bar=True,
)
```

### Example calculation for a DREAM.3D file

```python
import pygnd

# Calculate GND from DREAM.3D format
pygnd.calculate_and_save(
    dream3d_path="path/to/data.dream3d",
    grain_ids_path="DataContainers/ImageDataContainer/CellData/FeatureIds",
    euler_angles_path="DataContainers/ImageDataContainer/CellData/EulerAngles",
    cs=2,  # BCC
    burgers=2.48e-10,
    minimization="l2",
    slip_systems="screw+110",
    n_cpus=8,
    chunk_size=5000,
)
```

## Crystal Structures and Slip Systems

### FCC (cs=1)
- Slip systems: Always uses all slip systems

### BCC (cs=2)
Available slip system options:
- `"screw+110"` - Screw dislocations on {110} planes
- `"screw+112"` - Screw dislocations on {112} planes
- `"screw+123"` - Screw dislocations on {123} planes
- `"screw+110+112"` - Combined {110} and {112}
- `"screw+110+123"` - Combined {110} and {123}
- `"screw+112+123"` - Combined {112} and {123}
- `"all"` - All available slip systems

### HCP (cs=3)
Available slip system options:
- `"basal"` - Basal slip
- `"prismatic"` - Prismatic slip
- `"pyramidal"` - Pyramidal slip
- `"basal+prismatic"` - Combined basal and prismatic
- `"basal+pyramidal"` - Combined basal and pyramidal
- `"prismatic+pyramidal"` - Combined prismatic and pyramidal
- `"all"` - All available slip systems

## Parameters

### Main Function: `calculate_and_save()`

**Input file parameters:**
- `ang_path` (str, optional): Path to ANG file
- `dream3d_path` (str, optional): Path to DREAM.3D file
- `grain_ids_path` (str): Path to grain IDs data
- `euler_angles_path` (str, optional): Path to Euler angles (for DREAM.3D)

**Material parameters:**
- `cs` (int): Crystal structure (1=FCC, 2=BCC, 3=HCP)
- `burgers` (float): Burgers vector magnitude in meters
- `slip_systems` (str): Slip system selection (see above)

**Calculation parameters:**
- `minimization` (str or list): "l1", "l2", or both ["l1", "l2"]
- `n_cpus` (int): Number of CPU cores to use
- `chunk_size` (int): Data points per processing chunk
- `progress_bar` (bool): Show progress bar

## Output

The package saves results in `.npy` files by default. If a DREAM.3D file is used, it will attempt to add the output to the DREAM.3D file directly. In the case of an ANG file, images of the result are saved in addition to the `.npy` raw data. Raw data is saved in the following format:
- `gnd_l1.npy` or `gnd_l2.npy`: shape (n_slip_systems, Z, Y, X), to get the full GND density, sum across the first (n_slip_systems) axis.
- `fdm.npy`: shape (3, Z, Y, X), to get the average finite difference misorientation, take the mean along the first axis.

## Dependencies

- numpy >= 1.20
- scipy >= 1.7
- h5py >= 3.0
- matplotlib >= 3.3
- tqdm >= 4.50
- joblib >= 1.0

Conda environment example:

```
conda create -n pygnd_env python=3.12 numpy scipy h5py matplotlib tqdm joblib
```

## Testing

PyGND includes a comprehensive test suite with 72+ tests covering core functionality. To run tests:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run with coverage
pytest --cov=pygnd --cov-report=html
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Before submitting:

1. Ensure all tests pass: `pytest`
2. Add tests for new features
3. Follow the existing code style
4. Update documentation as needed

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Citation

If you use this code in your research, please cite the original papers:

1. Witzen, W.A., et al. (2020). "IN718 Paper". *International Journal of Plasticity*. [DOI: 10.1016/j.ijplas.2020.102709](https://doi.org/10.1016/j.ijplas.2020.102709)

2. Witzen, W.A., et al. (2022). "Spalled Ta Paper". *Acta Materialia*. [DOI: 10.1016/j.actamat.2022.118366](https://doi.org/10.1016/j.actamat.2022.118366)

3. Witzen, W.A., et al. (2022). "AM Ta Paper". *Journal of Materials Science*. [DOI: 10.1007/s10853-022-07074-2](https://doi.org/10.1007/s10853-022-07074-2)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **James Lamb** - Python implementation
- **Wyatt Witzen** - Original MATLAB implementation and methodology

## Acknowledgments

This code uses Nye's dislocation theory to relate orientation curvature tensors to geometrically necessary dislocation densities. The methodology was developed by Wyatt Witzen for the analysis of TriBeam 3D microstructures.
