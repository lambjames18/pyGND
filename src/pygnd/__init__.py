"""
PyGND - Geometrically Necessary Dislocation calculations for 3D microstructures.

This package provides tools for calculating geometrically necessary dislocation (GND)
densities from EBSD data using Nye's dislocation theory.
"""

from pygnd._version import __version__

from pygnd.core import calculate_and_save, get_linear_operator
from pygnd import io, rotations, quaternions, utils

__all__ = [
    "calculate_and_save",
    "get_linear_operator",
    "io",
    "rotations",
    "quaternions",
    "utils",
]
