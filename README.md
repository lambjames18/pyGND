# PyGND

### THIS CODE IS UNDER ACTIVE DEVELOPMENT

**Geometrically Necessary Dislocation calculations for TriBeam 3D microstructures**

[![Pylint](https://img.shields.io/endpoint?url=https%3A%2F%2Flambjames18.github.io%2FpyGND%2Fbadges%2Fpylint.json)](https://lambjames18.github.io/pyGND/badges/pylint.json)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Flambjames18.github.io%2FpyGND%2Fbadges%2Fcoverage.json)](https://lambjames18.github.io/pyGND/coverage/)
[![Docs](https://img.shields.io/badge/docs-pdoc-blue)](https://lambjames18.github.io/pyGND/api/)
[![CI](https://github.com/lambjames18/pyGND/actions/workflows/cicd.yml/badge.svg)](https://github.com/lambjames18/pyGND/actions/workflows/cicd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

PyGND is a Python package for calculating geometrically necessary dislocation (GND) densities from EBSD (Electron Backscatter Diffraction) data using Nye's dislocation theory. The code was originally developed in MATLAB by Wyatt Witzen and has been reimplemented in Python for improved performance and accessibility.

## Features

- **Multiple crystal structures**: FCC, BCC, and HCP
- **Flexible slip system selection**: Choose specific slip systems or use all available
- **Optimized calculations**: Parallel processing with configurable CPU cores
- **Multiple minimization methods**: L1 and L2 minimization
- **Multiple input formats**: Support for ANG files and DREAM.3D formats
- **Progress tracking**: Built-in progress bars for long calculations
- **Memory efficient**: Configurable chunking for large datasets
- **Desktop GUI** and a **command line interface**, in addition to the Python API

## Installation

### From PyPI

```bash
pip install pygnd
```

### From source

```bash
git clone https://github.com/lambjames18/pyGND.git
cd pyGND
pip install -e .
```

If you use [uv](https://docs.astral.sh/uv/) (the toolchain this project develops with), `uv sync` will set up an editable install with all dependencies resolved from `pyproject.toml`.

## Quick Start

### Python API

```python
import pygnd

# Calculate GND from an .ang file
pygnd.calculate_and_save_ang(
    "path/to/data.ang",
    cs=1,  # 1=FCC, 2=BCC, 3=HCP
    burgers=2.48e-10,  # Burgers vector magnitude in meters
    grain_ids_path="path/to/grain_data.txt",
    minimization=["l2", "l1"],
    slip_systems="all",
    n_cpus=4,
    progress_bar=True,
)

# Calculate GND from a DREAM.3D file and save the results back into it
pygnd.calculate_and_save_dream3d(
    "path/to/data.dream3d",
    ids_name="DataContainers/ImageDataContainer/CellData/FeatureIds",
    euler_name="DataContainers/ImageDataContainer/CellData/EulerAngles",
    cs=2,  # BCC
    burgers=2.48e-10,
    minimization="l2",
    slip_systems="screw+110",
    n_cpus=8,
    chunk_size=5000,
)
```

See [`examples/`](examples/) for complete runnable scripts, and the [API documentation](https://lambjames18.github.io/pyGND/api/) for the full parameter reference (including `pygnd.get_linear_operator`, the `pygnd.rotations`/`pygnd.quaternions` conversion utilities, and lower-level functions in `pygnd.core`).

### Command line

Installing the package also provides two console scripts:

```bash
# Run the same calculation as above from the command line
pygnd_calculate ang path/to/data.ang --cs 1 --burgers 2.48e-10 --grain-ids-path path/to/grain_data.txt
pygnd_calculate dream3d path/to/data.dream3d --ids-name FeatureIds --euler-name EulerAngles --cs 2 --burgers 2.48e-10

# Validate a file's shape/spacing before queuing a large job, without running the calculation
pygnd_calculate ang path/to/data.ang --dry-run

# Launch the desktop GUI
pygnd_gui
```

Run `pygnd_calculate --help`, `pygnd_calculate ang --help`, or `pygnd_calculate dream3d --help` for the full list of options (minimization scheme, CPU/chunk-size controls, slip systems, etc.) — these mirror the Python API parameters below.

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

For HCP with a mixed basal/prismatic + pyramidal slip-system combination, `burgers` must be a `(basal/prismatic, pyramidal)` tuple of the two Burgers vector magnitudes.

## Main Functions

- **`calculate_and_save_ang(ang_path, cs, burgers, grain_ids_path=None, ...)`** — calculate from an `.ang` file, saving results as `.npy` files and preview images next to it.
- **`calculate_and_save_dream3d(dream3d_path, ids_name, euler_name, cs, burgers, ...)`** — calculate from a DREAM.3D file, saving results back into it (falling back to `.npy` files if the DREAM.3D write fails).
- **`calculate_and_save(...)`** — a deprecated combined-argument entry point kept for backwards compatibility; prefer the two functions above.

Both functions share the following parameters:

- `cs` (int): Crystal structure (1=FCC, 2=BCC, 3=HCP)
- `burgers` (float or tuple): Burgers vector magnitude(s) in meters (see above for HCP mixed slip systems)
- `slip_systems` (str): Slip system selection (see above)
- `minimization` (str or list): `"l1"`, `"l2"`, or both `["l1", "l2"]`
- `n_cpus` (int): Number of CPU cores to use for L1 minimization (`-1` uses all available)
- `chunk_size` (int): Data points per processing chunk
- `progress_bar` (bool): Show a progress bar during L1 minimization

See the [API documentation](https://lambjames18.github.io/pyGND/api/) for the complete, per-function parameter reference.

## Output

The package saves results in `.npy` files by default. If a DREAM.3D file is used, it will attempt to add the output to the DREAM.3D file directly. In the case of an ANG file, images of the result are saved in addition to the `.npy` raw data. Raw data is saved in the following format:
- `gnd_l1.npy` or `gnd_l2.npy`: shape (n_slip_systems, Z, Y, X), to get the full GND density, sum across the first (n_slip_systems) axis.
- `fdm.npy`: shape (3, Z, Y, X), to get the average finite difference misorientation, take the mean along the first axis.

## Dependencies

Runtime dependencies (see `pyproject.toml` for the authoritative, unpinned list): `numpy`, `scipy`, `h5py`, `matplotlib`, `tqdm`, `joblib`.

Conda environment example:

```bash
conda create -n pygnd_env python=3.12 numpy scipy h5py matplotlib tqdm joblib
conda activate pygnd_env
pip install -e .
```

## Documentation

- [API documentation](https://lambjames18.github.io/pyGND/api/) (built with [pdoc](https://pdoc.dev/))
- [Test coverage report](https://lambjames18.github.io/pyGND/coverage/)

Both are rebuilt automatically on every push to `main`.

## Testing

PyGND has an extensive test suite covering all of `pygnd.core`, `pygnd.io`, `pygnd.quaternions`, `pygnd.rotations`, and the `pygnd_calculate` CLI — see the coverage badge/report above for current numbers.

```bash
# Install with dev dependencies (pytest, pytest-cov, pylint)
uv sync --all-extras --dev

# Run tests
uv run pytest

# Run with an HTML coverage report
uv run pytest --cov=pygnd --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Before submitting:

1. Ensure all tests pass: `uv run pytest`
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
