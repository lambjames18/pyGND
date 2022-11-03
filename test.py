import timeit
import numpy as np
import py_functions as pf

def deltathetakV5(gA, gB, k, symOp):
    numSym = symOp.shape
    misori_matrix = np.zeros(numSym[2]**2)
    gA_iter = 0
    gB_iter = 0
    # gA_temps = symOp.dot(gA.dot(np.transpose(symOp, axes=(1, 0, 2))))
    # gB_temps = symOp.dot(gB.dot(np.transpose(symOp, axes=(1, 0, 2))))
    # print((gA.dot(symOp.T)).shape, gA_temps.shape)

    if (gB != gA).any() and gA.sum() != 0:
        for delg_iter in range(misori_matrix.shape[0]):
            gA_temp = symOp[:, :, gA_iter].dot(gA.dot(symOp[:, :, gA_iter].T))
            gB_temp = symOp[:, :, gB_iter].dot(gB.dot(symOp[:, :, gB_iter].T))
            # gA_temp1 = gA_temps[:, :, gA_iter]
            # gB_temp1 = gB_temps[:, :, gB_iter]
            # print((gA_temp == gA_temp1).all())
            # print((gB_temp == gB_temp1).all())
            # continue
            delg = np.linalg.solve(gA_temp.conj().T, gB_temp.conj().T).conj().T
            deltheta = np.arccos((np.diag(delg).sum() - 1) / 2)
            
            if deltheta == 0:
                misori_matrix[delg_iter] = 0
            elif k == 1:
                misori_matrix[delg_iter] = -(delg[1, 2] - delg[2, 1]) * (deltheta/(2*np.sin(deltheta)))
            elif k == 2:
                misori_matrix[delg_iter] = -(delg[2, 0] - delg[0, 2]) * (deltheta/(2*np.sin(deltheta)))
            elif k == 3:
                misori_matrix[delg_iter] = -(delg[0, 1] - delg[1, 0]) * (deltheta/(2*np.sin(deltheta)))
            else:
                misori_matrix[delg_iter] = 0

            if gB_iter == numSym[2] - 1:
                gB_iter = 0
                gA_iter += 1
            else:
                gB_iter += 1

        d_col = np.argmin(np.abs(misori_matrix))
        disori = np.abs(misori_matrix[d_col])
    else:
        disori = 0
    return disori


symOp = pf.symmetry_operators(1)
symOp = symOp[:, :, :2]

# a = np.array([[1,2,3], [2,3,4], [5,6,7]], dtype=float)
# b = np.array([[3,2,1], [4,3,2], [7,6,5]], dtype=float)


a = pf.eu2om(np.array([10, 20, 30]))
b = pf.eu2om(np.array([12, 20, 25]))

aa = a.dot(np.transpose(symOp, axes=(1, 0, 2)))
print(aa.shape, symOp.shape)
aaa = np.einsum('abi,ibd->adi', symOp, np.transpose(aa, axes=(1, 0, 2)))
print(aaa.shape)
i = 1
print(aaa[:, :])
print()
print(symOp[:, :, 0].dot(a.dot(symOp[:, :, 0].T)))
print(symOp[:, :, 1].dot(a.dot(symOp[:, :, 1].T)))

# out0 = pf.deltathetakV4(a, b, 1, symOp)
out1 = deltathetakV5(a, b, 1, symOp)
# print(out0, out1)
# t = timeit.timeit("deltathetakV4(a, b, 1, symOp)", setup="from __main__ import deltathetakV4, a, b, symOp", number=1)
# print(t)

