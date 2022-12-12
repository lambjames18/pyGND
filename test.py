import timeit
import numpy as np
from skimage import io, exposure
import matplotlib.pyplot as plt


path = "C:/Users/lambj/Downloads/0485_Mod/0485_Mod/0485_Mod_x6950y1550.jpg"
img = io.imread(path, as_gray=True)

fig, ax = plt.subplots(1, 10, figsize=(30, 5))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")
for i in range(1, 10):
    clahe = exposure.equalize_adapthist(img, clip_limit=i/100)
    ax[i].imshow(clahe, cmap="gray")
    ax[i].set_title("CLAHE " + str(i/100))
plt.tight_layout()
plt.show()
