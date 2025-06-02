import matplotlib.pyplot as plt
import numpy as np
import scipy

from orix.quaternion import Orientation, Misorientation, OrientationRegion

import rotations
import quaternions as qpy
import utillities as ut


reference = "p"
p = (124, 194)
p = (171, 93)
p = (120, 70)
method = "refaxis"

path = [
    "E:/cells/Ortho_20240229_24197/CoNi90-OrthoCells_20240229_24197_scan3.ang",
    "E:/cells/CoNi90-OrthoCells_20240320_27061_scan3.ang",
    "E:/cells/CoNi90-ParallelCells_20240320_27064_scan6.ang",
    "E:/cells/Parallel_20240229_24207/CoNi90-ParallelCells_20240229_24207_scan6.ang",
][1]

eu, ids, s = ut.read_ang(path)
eu = eu[0]
ids = ids[0]
print("Scan size", ids.shape)

q = rotations.eu2qu(eu)
q = qpy.qu_norm_std(q)

angles = np.zeros_like(ids, dtype=float)
for ID in np.unique(ids):
    mask = ids == ID
    if ID == 0:
        continue
    print("ID:", ID, "N:", mask.sum())
    y, x = np.where(mask)
    if reference == "center":
        q0 = q[int(np.mean(y)), int(np.mean(x))]
    elif reference == "mean":
        q0 = qpy.qu_avg(q[mask], 11)
    elif reference == "p":
        q0 = q[p[0], p[1]]
    else:
        raise ValueError("Reference must be 'center' or 'mean'")

    q_dis = qpy.qu_disorientation_directional(q0, q[mask], 11)

    angles[mask] = qpy.get_sign_carrying_disorientation_angle(q_dis)

angles = np.rad2deg(angles)

vmin, vmax = np.percentile(angles, [1, 99])
vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
fig, ax = ut.view(angles, "Angle", "Spectral_r", vmin=vmin, vmax=vmax, show=False)
ax.scatter(p[1], p[0], c="red", marker="x", s=50)
plt.show()
