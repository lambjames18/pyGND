# GND calculations in a ANG file
# Author: James Lamb (GND calculations originally from Wyatt Witzen)


import pygnd


#############################################
# Path to the ANG file
path = "../demo_data/CoNi.ang"

# Name of the DataArray containing the Feature IDs
grain_ids_path = "../demo_data/CoNi_grain_data.txt"

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
#############################################


pygnd.calculate_and_save_ang(
    path,
    cs=cs,
    burgers=burgers,
    grain_ids_path=grain_ids_path,
    minimization=minimization,
    slip_systems=slip_systems,
    n_cpus=n_cpus,
    progress_bar=progress_bar,
    chunk_size=chunk_size,
)
