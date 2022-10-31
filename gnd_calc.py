import numpy as np

import py_functions as pf

directory = "./test/"
# Name output file
ID = "Test_Output"
# Get crystallography
burgers, A_sparse, numModes, cs = pf.xtal()

# Convert Burgers to m
burgers = burgers * 1e-10

# Get symmetry operations from crystallography
symOp = pf.symmetry_operators(cs)

# Read data
### NEED TO DO THIS ###
