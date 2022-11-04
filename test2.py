import timeit
import numpy as np
import py_functions as pf


def slow(a, b, symOp):
    dthe = np.zeros((3, 3))
    dthe[0, 0] = pf.deltathetakV4(a, b1, 1, symOp)
    dthe[1, 0] = pf.deltathetakV4(a, b1, 2, symOp)
    dthe[2, 0] = pf.deltathetakV4(a, b1, 3, symOp)
    dthe[0, 1] = pf.deltathetakV4(a, b2, 1, symOp)
    dthe[1, 1] = pf.deltathetakV4(a, b2, 2, symOp)
    dthe[2, 1] = pf.deltathetakV4(a, b2, 3, symOp)
    dthe[0, 2] = pf.deltathetakV4(a, b3, 1, symOp)
    dthe[1, 2] = pf.deltathetakV4(a, b3, 2, symOp)
    dthe[2, 2] = pf.deltathetakV4(a, b3, 3, symOp)
    return dthe

def fast(a, b, symOp):
    dthe = np.zeros((3, 3))

symOp = pf.symmetry_operators(1)

a = pf.eu2om(np.array([10, 20, 30]))
b1 = pf.eu2om(np.array([12, 20, 35]))
b2 = pf.eu2om(np.array([ 8, 20, 25]))
b3 = pf.eu2om(np.array([15, 25, 30]))

