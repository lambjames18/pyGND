# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# Designed to read in and write out to DREAM3D files, but can be adapted for other formats


import pygnd


#############################################
# Path to the DREAM3D file
path = "../demo_data/CoNi.dream3d"

# Name of the DataArray containing the Feature IDs
ids_name = "FeatureIds"

# Name of the DataArray containing the Euler angles
euler_name = "EulerAngles"

# Burgers vector magnitude in m
burgers = 2.48e-10

# Number of CPU cores to use
n_cpus = 5

# Crystal structure, 1 = FCC, 2 = BCC, 3 = HCP
cs = 1

# Slip systems
# (FCC) - unused, always 'all'
# (BCC) - 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'screw+110+123', 'screw+112+123', 'all'
# (HCP) - 'basal', 'prismatic', 'pyramidal', 'basal+prismatic', 'basal+pyramidal', 'prismatic+pyramidal', 'all'
slip_systems = "all"

# "l2" or "l1" (where l1 is the absolute value); l2 is faster, l1 may be more accurate
# can be a list/tuple of both as well, e.g. minimization = ["l1", "l2"]
minimization = ["l2", "l1"]

# Whether to show a progress bar
progress_bar = True

# How many data points to process in one chunk (decrease if memory issues)
chunk_size = 1000

# Spacing units, ensures that the spacing is converted to meters correctly
units = "um"
#############################################


pygnd.calculate_and_save(
    dream3d_path=path,
    ids_name=ids_name,
    euler_name=euler_name,
    spacing_units=units,
    cs=cs,
    burgers=burgers,
    minimization=minimization,
    slip_systems=slip_systems,
    n_cpus=n_cpus,
    progress_bar=progress_bar,
    chunk_size=chunk_size,
)
