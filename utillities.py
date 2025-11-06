import contextlib
import joblib

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py


def read_ang(path, ids_path=None):
    """Reads an ang file into a numpy array"""
    num_header_lines = 0
    col_names = None
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#":
                num_header_lines += 1
                if "NCOLS_ODD" in line:
                    ncols = int(line.split(": ")[1].strip())
                elif "NROWS" in line:
                    nrows = int(line.split(": ")[1].strip())
                elif "COLUMN_HEADERS" in line:
                    col_names = line.split(": ")[1].strip().split(", ")
                elif "XSTEP" in line:
                    res = float(line.split(": ")[1].strip())
            else:
                break
    if col_names is None:
        col_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]
    raw_data = np.genfromtxt(path, skip_header=num_header_lines)
    n_entries = raw_data.shape[-1]
    if raw_data.shape[0] == ncols * nrows:
        data = raw_data.reshape((nrows, ncols, n_entries))
    elif raw_data.shape != ncols * nrows:
        raise ValueError(
            f"The number of data points ({raw_data.size}) does not match the expected grid ({nrows} rows, {ncols} cols, {ncols * nrows} total points). "
        )

    out = {col_names[i]: data[:, :, i] for i in range(n_entries)}
    eulerangles = np.array([out["phi1"], out["PHI"], out["phi2"]]).T.astype(float)
    eulerangles = eulerangles.reshape(1, *eulerangles.shape).transpose(0, 2, 1, 3)
    if ids_path is not None:
        grain_data = np.genfromtxt(ids_path, dtype=float, comments="#")
        ids = grain_data[:, 8].reshape(eulerangles.shape[:-1]).astype(int)
        # ids = np.load(ids_path).reshape(eulerangles.shape[:-1])
    else:
        ids = np.ones(eulerangles.shape[:-1], dtype=int)
    spacing = np.array([res, res, res])
    return eulerangles, ids, spacing


def read_dream3d(
    path: str,
    ids_name: str = "FeatureIds",
    euler_name: str = "EulerAngles",
    spacing_units: str = "microns",
) -> tuple:
    """Reads a dream3d file into a numpy array"""

    ids = np.squeeze(extract_data_from_h5(path, ids_name))
    if ids is None:
        raise KeyError(
            f"Could not find a data array with the name '{ids_name}' in the dream3d file."
        )

    eulerangles = extract_data_from_h5(path, euler_name)
    if eulerangles is None:
        raise KeyError(
            f"Could not find a data array with the name '{euler_name}' in the dream3d file."
        )

    spacing = read_dream3d_spacing(path, spacing_units=spacing_units)

    return eulerangles, ids, spacing


def extract_path_from_h5(h5_file_path, target_name):
    with h5py.File(h5_file_path, "r") as h5:

        def recursive_search(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name.endswith(target_name):
                    return name
                else:
                    return None
            for key, item in obj.items():
                result = recursive_search(f"{name}/{key}", item)
                if result:
                    return result
            return None

        path = recursive_search("", h5)
    return path


def extract_attribute_from_h5(h5_file_path, attribute_name):
    with h5py.File(h5_file_path, "r") as h5:

        def recursive_search(name, obj):
            if isinstance(obj, h5py.Group):
                if attribute_name in obj.attrs:
                    return obj.attrs[attribute_name]
            for key, item in obj.items():
                result = recursive_search(f"{name}/{key}", item)
                if result is not None:
                    return result
            return None

        attribute_value = recursive_search("", h5)
    return attribute_value


def extract_data_from_h5(h5_file_path, target_name):
    with h5py.File(h5_file_path, "r") as h5:

        def recursive_search(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name.endswith(target_name):
                    return obj[...]
                else:
                    return None
            for key, item in obj.items():
                result = recursive_search(f"{name}/{key}", item)
                if result is not None:
                    return result
            return None

        data_array = recursive_search("", h5)
    return data_array


def read_dream3d_spacing(path: str, spacing_units: str = "microns") -> np.ndarray:
    """Read the spacing from a dream3d file."""
    spacing = extract_data_from_h5(path, "SPACING")

    # If the spacing path isn't found, it is likely a dream3dnx file and spacing is an attribute
    if spacing is None:
        spacing = extract_attribute_from_h5(path, "_SPACING")
        if spacing is None:
            raise ValueError(
                "Could not find spacing information in the dream3d file. Attempted to find a 'SPACING' dataset (DREAM3D format) and a '_SPACING' attribute (DREAM3DNX format)."
            )

    # Convert spacing to meters
    if (
        spacing_units == "nm"
        or spacing_units == "nanometer"
        or spacing_units == "nanometers"
    ):
        spacing *= 1e-9
    elif (
        spacing_units == "um"
        or spacing_units == "micron"
        or spacing_units == "microns"
        or spacing_units == "micrometer"
        or spacing_units == "micrometers"
        or "µm"
    ):
        spacing *= 1e-6
    elif (
        spacing_units == "mm"
        or spacing_units == "millimeter"
        or spacing_units == "millimeters"
    ):
        spacing *= 1e-3
    elif spacing_units == "m" or spacing_units == "meter" or spacing_units == "meters":
        pass
    else:
        raise ValueError(
            "units must be one of 'nm', 'nanometer', 'um', 'µm', 'micron', 'microns', 'micrometer', 'micrometers', 'mm', 'millimeter', 'millimeters', 'm', or 'meters'."
        )

    return spacing


def standardize_axis(ax, grid=True, **kwargs):

    kwargs["labelsize"] = kwargs.get("labelsize", 20)
    kwargs["labelcolor"] = kwargs.get("labelcolor", "k")
    kwargs["direction"] = kwargs.get("direction", "in")
    kwargs["top"] = kwargs.get("top", True)
    kwargs["right"] = kwargs.get("right", True)
    ax.tick_params(axis="both", which="both", **kwargs)
    if grid:
        ax.grid(alpha=0.3, which="major")
        ax.grid(alpha=0.1, which="minor")
        ax.set_axisbelow(True)


def make_legend(ax, **kwargs):
    kwargs["bbox_to_anchor"] = kwargs.get("bbox_to_anchor", (1.03, 1.05))
    kwargs["loc"] = kwargs.get("loc", "upper right")
    kwargs["fontsize"] = kwargs.get("fontsize", 15)
    kwargs["shadow"] = kwargs.get("shadow", True)
    kwargs["framealpha"] = kwargs.get("framealpha", 1)
    kwargs["fancybox"] = kwargs.get("fancybox", False)
    ax.legend(**kwargs)


def make_axis_log(ax, axis="x"):
    """Make an axis logarithmic."""
    if axis in ["x", "both"]:
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        xmin, xmax = ax.get_xlim()
        tick_range = np.arange(np.ceil(xmin), np.floor(xmax) + 1)
        tick_range_minor = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
        ax.xaxis.set_ticks(tick_range)
        minor_ticks = []
        for p in tick_range_minor:
            for x in np.linspace(10**p, 10 ** (p + 1), 10):
                if np.log10(x) >= xmin and np.log10(x) <= xmax:
                    minor_ticks.append(np.log10(x))
        ax.xaxis.set_ticks(minor_ticks, minor=True)
    if axis in ["y", "both"]:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ymin, ymax = ax.get_ylim()
        tick_range = np.arange(np.ceil(ymin), np.floor(ymax) + 1)
        tick_range_minor = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
        ax.yaxis.set_ticks(tick_range)
        minor_ticks = []
        for p in tick_range_minor:
            for y in np.linspace(10**p, 10 ** (p + 1), 10):
                if np.log10(y) >= ymin and np.log10(y) <= ymax:
                    minor_ticks.append(np.log10(y))
        ax.yaxis.set_ticks(minor_ticks, minor=True)


def view(arr, title, cmap, vmin=None, vmax=None, log=False, show=True, return_ax=True):
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
    fig, ax = plt.subplots(1, 1, figsize=(3.9, 3), dpi=300)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.78, top=0.98, bottom=0.02)
    l = ax.get_position()
    cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    fig.colorbar(im, cax=cax, label=title)
    if log:
        make_axis_log(cax, "y")
    if show:
        plt.show()
    if return_ax:
        return fig, ax


def view_simple(arr, cmap, save_path=None, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
    size = (arr.shape[1] / 300, arr.shape[0] / 300)
    fig, ax = plt.subplots(1, 1, figsize=size, dpi=300)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


# Context manager to patch joblib to report into tqdm progress bar given as argument
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


if __name__ == "__main__":

    path = "/Users/jameslamb/Downloads/Test.dream3d"
    spacing = read_dream3d_spacing(path)
    print("Spacing:", spacing)

    path = "/Users/jameslamb/Documents/research/data/IN718SS_Justine/Testing_Cropped_IN718SS.dream3d"
    spacing = read_dream3d_spacing(path)
    print("Spacing:", spacing)
