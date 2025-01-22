import matplotlib.pyplot as plt
import numpy as np
import scipy

from orix.quaternion import Orientation, Misorientation, OrientationRegion

import rotations
import quaternions as qpy
import utillities as ut


eu, ids, s = ut.read_ang("E:/rolled_al/merged_1x1.ang", "E:/rolled_al/FeatureIDs.npy")
eu = eu[0]
ids = ids[0]
# eu = eu[0, :100, :100]
q = rotations.eu2qu(eu)
q = qpy.qu_norm_std(q)

angles = np.zeros_like(ids, dtype=float)
for ID in np.unique(ids):
    mask = ids == ID
    if ID == 0:
        continue
    y, x = np.where(mask)
    q0 = q[int(np.mean(y) + 1), int(np.mean(x) + 1)]
    dis = qpy.qu_disorientation(q0, q[mask], 11, 11)
    axes = qpy.qu_axis(dis)
    a = qpy.qu_angle(dis)
    a[axes[..., 2] < 0] *= -1
    angles[mask] = np.rad2deg(a)
    
vmin, vmax = angles.min(), angles.max()
print(vmin, vmax)
vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)

ut.view(angles, "Angle", 'Spectral_r', vmin=vmin, vmax=vmax)

