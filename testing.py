import matplotlib.pyplot as plt
import numpy as np
import time
import memory_profiler
import h5py

from GND import get_finite_difference_coordinates


def main(ids):
    # process the array
    get_finite_difference_coordinates(ids)


if __name__ == "__main__":
    interval = 0.01
    iterations = 3
    sizes = [20, 50, 100, 200, 500]

    h5 = h5py.File("/Users/jameslamb/Documents/research/data/CoNi90/CoNi90.dream3d", 'r')
    ids = h5["DataContainers/ImageDataContainer/CellData/FeatureIds"][..., 0]
    h5.close()

    c = np.array(ids.shape) // 2
    ids_sized = []
    for i in range(len(sizes)):
        x = slice(max(0, c[0]-sizes[i]//2), min(ids.shape[0], c[0]+sizes[i]//2))
        y = slice(max(0, c[1]-sizes[i]//2), min(ids.shape[1], c[1]+sizes[i]//2))
        z = slice(max(0, c[2]-sizes[i]//2), min(ids.shape[2], c[2]+sizes[i]//2))
        ids_sized.append(ids[x, y, z])

    usage = []
    times = []
    for i in range(len(ids_sized)):
        print(f"Running test for shape {ids_sized[i].shape}")
        elapsed = []
        used = []
        for _ in range(iterations):
            t0 = time.time()
            mem_usage = memory_profiler.memory_usage((main, (ids_sized[i],)), interval=interval)
            t1 = time.time()
            elapsed.append(t1 - t0)
            used.append(max(mem_usage))
        times.append((np.mean(elapsed), np.std(elapsed)))
        usage.append((np.mean(used), np.std(used)))
    
    num_elements = [np.prod(s.shape) for s in ids_sized]
    tm = [t[0] for t in times]
    um = [u[0] for u in usage]
    ts = [t[1] for t in times]
    us = [u[1] for u in usage]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].errorbar(num_elements, tm, yerr=ts, fmt='o-')
    ax[0].set_ylabel("Time (s)")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title("Time vs. number of elements")
    ax[0].grid()

    ax[1].errorbar(num_elements, um, yerr=us, fmt='o-', label="Memory usage")
    ax[1].set_ylim(10, max(um) * 1.1)
    ax[1].set_xlabel("Number of elements")
    ax[1].set_ylabel("Memory usage (MB)")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title("Memory usage vs. number of elements")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()
        


