import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from utillities import extract_path_from_h5


######################################
# Path to either a npy or dream3d file
path = ...

# Take the log10 of the data for visualization (recommended for GND densities)
log = True

# Colormap for visualization
cmap = "RdBu_r"

# Minimum and maximum values for visualization (set to None for automatic)
vmin = None
vmax = None
######################################


class InteractiveSlider:
    def __init__(self, stack, alphas=None, vmin=None, vmax=None, cmap="gray"):
        if alphas is None:
            alphas = np.ones(stack.shape[1:3])
        if vmin is None:
            vmin = np.min(stack)
        if vmax is None:
            vmax = np.max(stack)
        self.stack = stack
        self.vmin = vmin
        self.vmax = vmax
        self.alphas = alphas
        self.cmap = cmap
        plt.close(81234)
        self.fig = plt.figure(81234, figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("")
        # Show images
        self.im = self.ax.imshow(
            self.stack[0],
            alpha=self.alphas,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=self.cmap,
        )
        # Put slider on
        plt.subplots_adjust(left=0.15, bottom=0.15)
        left = self.ax.get_position().x0
        bot = self.ax.get_position().y0
        height = self.ax.get_position().height
        axslice = plt.axes([left - 0.15, bot, 0.05, height])
        self.slice_slider = Slider(
            ax=axslice,
            label="Slice #",
            valmin=0,
            valmax=len(stack) - 1,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )

        # Enable update functions
        self.slice_slider.on_changed(self.update_slice)
        plt.show()

    def update_slice(self, val):
        val = int(np.around(val, 0))
        image = self.stack[val]
        self.im.set_data(image)
        self.im.axes.figure.canvas.draw()
        self.fig.canvas.draw_idle()


# Create a function that parses an HDF5 file and extracts the string path to a data array titled "GND"


if path.endswith(".dream3d"):
    gnd_path = extract_path_from_h5(path, "GND")
    print("GND data path:", gnd_path)
    h5 = h5py.File(path, "r")
    data = h5[gnd_path][..., 0]
    h5.close()

if path.endswith(".npy"):
    data = np.load(path)
    data = data.sum(axis=0)

if log:
    data = np.log10(data, where=data > 0)

non_zero = data[data > 0]
mn, mx, mean, std = non_zero.min(), non_zero.max(), non_zero.mean(), non_zero.std()

print("Data Summary:")
print("Data shape:", data.shape)
print("Max:", mx)
print("Non-zero min:", mn)
print("Mean:", mean)
print("Std:", std)

vmin = vmin if vmin is not None else mn
vmax = vmax if vmax is not None else mx

print(f"Visualization range: vmin={vmin}, vmax={vmax}")

InteractiveSlider(data, vmin=vmin, vmax=vmax, cmap=cmap)
