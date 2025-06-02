import h5py
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

Oh = symmetry.Oh

import quaternions as qpy
import rotations
import utillities as ut


ang_path = "/Users/jameslamb/Documents/research/data/coni/DED_CoNi90.ang"
ids_path = "/Users/jameslamb/Documents/research/data/coni/DED_CoNi90.txt"

slc = (slice(750, 850), slice(450, 550))

eu, _, s = ut.read_ang(ang_path)
q = rotations.eu2qu(eu)[0].transpose(1, 0, 2)[::-1][slc]
center = np.array(q.shape[:-1]) // 2

o = Orientation(q, symmetry=Oh)
ipf = IPFColorKeyTSL(Oh, Vector3d((1, 0, 0))).orientation2color(o)

mis = qpy.qu_disorientation_directional(q[26, 56], q, 11)
mis = qpy.get_sign_carrying_disorientation_angle(
    mis.reshape(-1, 4), chunk_size=1000  # , r_star=np.array([0, 0, 1])
).reshape(q.shape[:2])
# mis = qpy.qu_angle(mis)
print(mis.max())

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(ipf)
ax[1].imshow(mis * 180 / np.pi, vmin=0, vmax=3, cmap="Spectral_r")
plt.show()
