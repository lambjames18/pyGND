"""Command line interface for running GND calculations from a terminal or an
HPC batch script, without needing to write a Python driver script.
"""

import argparse
import sys

import numpy as np

from pygnd import __version__, io
from pygnd.core import calculate_and_save_ang, calculate_and_save_dream3d


_SLIP_SYSTEMS_HELP = (
    "Slip systems to use. 'all' works for every crystal structure. BCC also "
    "accepts: screw+110, screw+112, screw+123, screw+110+112, screw+110+123, "
    "screw+112+123. HCP also accepts: basal, prismatic, pyramidal, "
    "basal+prismatic, basal+pyramidal, prismatic+pyramidal. (default: %(default)s)"
)

_LOGO = """
                                                                                            
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                   ---------      -----+     ######   ##########        
                                ---------------   -----++    ######   ###############   
                               --------------     -----+++   ######   ################# 
      ###                     --------   ---      -----++++  ######   ###### +##########
############   ####     #### -------  ---------   -----++-+- ######   ######     #######
#####    ####   ####   ####  -------  ---------   -----+---+++#####   ######     +######
####      ####   ###   ###   -------  ---------   -------+++++#####   ######     #######
####      ####   #### ####    --------   ------   ------ -++++#####   ########+######## 
####     ####     #######      ----------------   ------  +++######   ################  
############       #####        ---------------   ------   +++#####   ###############   
#### #####          ###             --------      ------    ++#####   #########         
####               ####                                                                 
####            ######                                                                  
####           #####                                                                    
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
                                                                                        
"""



def _parse_burgers(value: str):
    """Parse a --burgers argument into a float, or a tuple of two floats.

    Args:
        value: a single number, or two comma-separated numbers
            (`basal/prismatic,pyramidal`) for mixed HCP slip systems.

    Returns:
        A `float`, or a `tuple[float, float]` for the two-value form.
    """
    parts = [p for p in value.split(",") if p.strip()]
    try:
        floats = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse burgers vector {value!r}") from exc
    if len(floats) == 1:
        return floats[0]
    if len(floats) == 2:
        return tuple(floats)
    raise argparse.ArgumentTypeError(
        "--burgers must be a single value, or two comma-separated values "
        "(basal/prismatic,pyramidal) for mixed HCP slip systems"
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the calculation arguments shared by the dream3d and ang subcommands."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and validate the input file (report array shapes and "
        "exit) without running the GND calculation. --cs and --burgers are "
        "not required in this mode.",
    )
    parser.add_argument(
        "--cs",
        type=int,
        default=None,
        choices=(1, 2, 3),
        help="Crystal structure: 1 for FCC, 2 for BCC, 3 for HCP. "
        "Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--burgers",
        type=_parse_burgers,
        default=None,
        metavar="VALUE[,VALUE]",
        help="Burgers vector magnitude in meters. For HCP with mixed "
        "basal/prismatic + pyramidal slip systems, pass two comma-separated "
        "values, e.g. 2.48e-10,2.5e-10. Required unless --dry-run is set.",
    )
    parser.add_argument("--slip-systems", default="all", help=_SLIP_SYSTEMS_HELP)
    parser.add_argument(
        "--minimization",
        nargs="+",
        default=["l2"],
        choices=("l1", "l2"),
        help="Minimization scheme(s) to use. Pass both with --minimization l1 l2. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=-1,
        help="Number of CPUs to use for parallel processing during L1 "
        "minimization; -1 uses all available cores. Not used for L2. (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of voxels to process per chunk during parallel L1 "
        "minimization. (default: %(default)s)",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Display a progress bar during L1 minimization.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the `pygnd_calculate` console script."""
    parser = argparse.ArgumentParser(
        prog="pygnd_calculate",
        description="Calculate geometrically necessary dislocation (GND) "
        "densities from an EBSD dataset and save the results.",
    )
    parser.add_argument("--version", action="version", version=f"pygnd {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dream3d = subparsers.add_parser(
        "dream3d",
        help="Calculate from a DREAM3D file and save the results back into it.",
    )
    dream3d.add_argument("dream3d_path", help="Path to the DREAM3D file.")
    dream3d.add_argument(
        "--ids-name", required=True, help="Name of the grain ID data array in the DREAM3D file."
    )
    dream3d.add_argument(
        "--euler-name",
        required=True,
        help="Name of the Euler angles data array in the DREAM3D file.",
    )
    dream3d.add_argument(
        "--spacing-units",
        default="um",
        help="Units of the voxel spacing stored in the DREAM3D file. (default: %(default)s)",
    )
    _add_common_arguments(dream3d)

    ang = subparsers.add_parser(
        "ang", help="Calculate from an .ang file and save the results as .npy files."
    )
    ang.add_argument("ang_path", help="Path to the .ang file.")
    ang.add_argument(
        "--grain-ids-path",
        default=None,
        help="Path to a grain-ID file generated by OIM Analysis. If omitted, "
        "the entire dataset is treated as a single grain.",
    )
    _add_common_arguments(ang)

    return parser


def _dry_run(args: argparse.Namespace) -> int:
    """Load the input file and report array shapes without running the GND
    calculation. Useful as a fast pre-flight check before queuing a large HPC
    job, to catch a bad file path or dataset name early.

    Args:
        args: parsed command line arguments.

    Returns:
        Process exit code: `0` if the file loaded successfully, `1` otherwise.
    """
    try:
        if args.command == "dream3d":
            euler, ids, spacing = io.read_dream3d(
                args.dream3d_path, args.ids_name, args.euler_name, args.spacing_units
            )
        else:
            euler, ids, spacing = io.read_ang(args.ang_path, args.grain_ids_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Dry run: file loaded successfully, no calculation performed.")
    print("Euler angles shape:", euler.shape)
    print("Grain IDs shape:", ids.shape)
    print("Voxel spacing (m):", spacing)
    print("Unique grain IDs:", np.unique(ids).size)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `pygnd_calculate` console script.

    Args:
        argv: command line arguments, or `None` to use `sys.argv`.

    Returns:
        Process exit code: `0` on success, `1` on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        return _dry_run(args)

    if args.cs is None or args.burgers is None:
        parser.error("--cs and --burgers are required unless --dry-run is set")

    minimization = (
        args.minimization[0] if len(args.minimization) == 1 else tuple(args.minimization)
    )

    try:
        if args.command == "dream3d":
            success = calculate_and_save_dream3d(
                args.dream3d_path,
                args.ids_name,
                args.euler_name,
                args.cs,
                args.burgers,
                spacing_units=args.spacing_units,
                minimization=minimization,
                n_cpus=args.n_cpus,
                slip_systems=args.slip_systems,
                progress_bar=args.progress_bar,
                chunk_size=args.chunk_size,
            )
        else:
            success = calculate_and_save_ang(
                args.ang_path,
                args.cs,
                args.burgers,
                grain_ids_path=args.grain_ids_path,
                minimization=minimization,
                n_cpus=args.n_cpus,
                slip_systems=args.slip_systems,
                progress_bar=args.progress_bar,
                chunk_size=args.chunk_size,
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0 if success else 1


_ENTRY_POINTS = [
    ("pygnd", "Show this summary (version, logo, available entry points)."),
    (
        "pygnd_calculate",
        "Run GND calculations from the command line. Subcommands: dream3d, ang.",
    ),
    ("pygnd_gui", "Launch the desktop GUI."),
]


def _build_info_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the `pygnd` console script."""
    parser = argparse.ArgumentParser(
        prog="pygnd",
        description="Show the PyGND version and available command line entry points.",
    )
    parser.add_argument("--version", action="version", version=f"pygnd {__version__}")
    return parser


def info(argv: list[str] | None = None) -> int:
    """Entry point for the `pygnd` console script: prints the logo, the
    installed version, and a summary of the other command line entry points
    this package provides.

    Args:
        argv: command line arguments, or `None` to use `sys.argv`.

    Returns:
        Process exit code: always `0`.
    """
    _build_info_parser().parse_args(argv)

    logo_lines = _LOGO.splitlines()
    while logo_lines and not logo_lines[0].strip():
        logo_lines.pop(0)
    while logo_lines and not logo_lines[-1].strip():
        logo_lines.pop()
    print("\n".join(logo_lines))
    print()
    print(f"pygnd {__version__}")
    print()
    print("Available command line entry points:")
    for name, description in _ENTRY_POINTS:
        print(f"  {name:<16} {description}")
    print()
    print("Run any entry point with --help for its full list of options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
