"Project cleaned data using UMAP."

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

n_cpus = int(os.cpu_count() / 2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d  [%(levelname)-10s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    load_dotenv()
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    assert os.path.isfile(JSON_PATH), "Please configure .env to point to params.json"
    with open(JSON_PATH, "r") as f:
        params_json = json.load(f)

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=params_json["data_set"],
        help="Specify the data set.",
    )

    parser.add_argument(
        "-M",
        "--homology_max_dim",
        default=params_json["homology_max_dim"],
        type=int,
        help="Maximum homology dimension for Ripser.py",
    )

    parser.add_argument(
        "-f",
        "--filter",
        default=params_json["filtration"],
        type=str,
        help="Filtration abbreviation, e.g. `rips` for Vietoris Rips.",
    )

    parser.add_argument(
        "-s",
        "--subset",
        default=None,
        nargs="+",
        help="Choose a subset of projections to compute homology.",
    )

    parser.add_argument(
        "-v",
        "--Verbose",
        default=False,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    # Write Coordinates to JSON

    # Subprocesses
    num_loops = len(params_json["coordinates"])

    ### Generate Diagrams

    logging.info(f"Computing Persistent Homology with `{args.filter}` Filtration")
    logging.info(f"Number of diagrams to generate: {num_loops}")
    logging.info(f"Maximum Homology Dim: {args.homology_max_dim}")

    # if args.subset is not None:
    #     logging.info(f"Interval Subset: {args.subset}")
    #     parameter_space = args.subset

    homology = os.path.join(root, "src/homology.py")
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for i in range(len(params_json["coordinates"])):
        cmd = [
            "python",
            f"{homology}",
            f"-i {i}",
            f"-M {args.homology_max_dim}",
        ]
        subprocesses.append(cmd)

    # Running processes in Parallel
    # TODO: optimize based on max_workers
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses
        progress_bar = tqdm(total=num_loops, desc="Progress", unit="subprocess")
        for future in as_completed(futures):
            # Update the progress bar for each completed subprocess
            progress_bar.update(1)
    progress_bar.close()
