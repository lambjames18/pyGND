
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py


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
        self.im = self.ax.imshow(self.stack[0], alpha=self.alphas, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
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

# name = "R2S10S5"
# name = "CoNiS29S2"
# name = "R2S9S4"
# name = "CoNi16"
# name = "CoNi90"
# sr = np.load(f"./output_data/{name}_GND_SR.npy")
# ss = np.load(f"./output_data/{name}_GND_SS.npy")
# ms = np.load(f"./output_data/{name}_misori.npy")
sr = np.load(f"D:/Research/Ta_AM-Spalled/Data/3D/Ta-AM_Spalled_GND_SR.npy")

sr = np.log10(sr, where=sr > 0)

print(sr.shape)
# sr = np.swapaxes(sr, 1, 0)

# sr_data = np.zeros((200, sr.shape[0]))
# for i in range(sr.shape[0]):
#     hist, edges = np.histogram(sr[i, sr[i] > 0], bins=200, range=(13, 17), density=True)
#     sr_data[:, i] = hist[::-1]
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# im = ax.imshow(sr_data, cmap="jet")
# ax.set_xlabel("Slice #")
# ax.set_yticks(np.linspace(0, 200, 5))
# ax.set_yticklabels(np.linspace(17, 13, 5))
# 
# cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
# plt.colorbar(im, cax=cax)
# plt.show()
# exit()

mn, mx = np.percentile(sr[sr > 0], (2.0, 95.0))
# mn, mx = sr[sr > 0].min(), sr[sr > 0].max()
print(mn, mx)
InteractiveSlider(sr, vmin=mn, vmax=mx, cmap="jet")
