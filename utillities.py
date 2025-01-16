import numpy as np


def read_ang(path):
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
        raise ValueError(f"The number of data points ({raw_data.size}) does not match the expected grid ({nrows} rows, {ncols} cols, {ncols * nrows} total points). ")
        
    out = {col_names[i]: data[:, :, i] for i in range(n_entries)}
    eulerangles = np.array([out["phi1"], out["PHI"], out["phi2"]]).T.astype(float)
    featIDs = np.ones(eulerangles.shape[:-1], dtype=int)
    eulerangles = eulerangles.reshape(1, *eulerangles.shape)
    featIDs = featIDs.reshape(1, *featIDs.shape)
    spacing = np.array([res, res, res])
    return eulerangles, featIDs, spacing
