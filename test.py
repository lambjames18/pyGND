import timeit
import numpy as np
import py_functions as pf
from scipy.spatial.transform import Rotation as R

ae = np.array([142.8, 32.0, 214.4])
ao = pf.eu2om(ae)
print(ao)

