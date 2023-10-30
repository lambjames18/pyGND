# TriBeam_GND
GND code for use with TriBeam 3D Microstructures. The code was originally developed in Matlab by Wyatt Witzen (see [IN718 Paper](https://doi.org/10.1016/j.ijplas.2020.102709), [Spalled Ta Paper](https://doi.org/10.1016/j.actamat.2022.118366), [AM Ta Paper](https://doi.org/10.1007/s10853-022-07074-2)), but this newer implementation is written in python. In short, it uses dislocation theory proposed by Nye to relate a orientation curvature tensor to geometrically necessary dislocation densities.

`GND.py` is the brain of the code and houses the GND calculations. It contains a class `GND` that houses the needed data with methods for running the calculations.

`TriBeam_run.py` is a script for running the GND calculations on a TriBeam dataset. This is function is to be run in the command line with a config file that passes the data file and other relevant information for the calculations. This script was put together in order to be ran in a HPC type environment (although it works just fine on any Windows, Mac, or Linux machine).

There are a couple other files that are usefuly for looking at the output and putting the output into the DREAM.3D file (for further analysis of a TriBeam dataset).
