"Project cleaned data using UMAP."

import argparse
import json
import logging
import os
import subprocess
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d  [%(levelname)-10s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    from utils import parameter_coordinates

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
        "-n",
        "--num_samples",
        default=params_json["num_samples"],
        type=int,
        help="Set number of samples in data set",
    )
    parser.add_argument(
        "-c",
        "--num_clusters",
        default=params_json["num_clusters"],
        type=int,
        help="Set number of artifical clusters data set",
    )
    parser.add_argument(
        "--projector",
        type=str,
        default=params_json["projector"],
        help="Set to the name of projector for dimensionality reduction. ",
    )

    parser.add_argument(
        "--hyperparams",
        default=params_json["hyperparams"],
        type=dict,
        help="Hyper parameters",
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

    logging.info(f"Data set: {args.data}")
    logging.info(f"num_samples: {len(args.data)}")
    logging.info(f"Choice of Projector: {args.projector}")
    logging.info(f"Hyperparameters: {args.hyperparams}")

    # Write Coordinates to JSON
    parameter_space = parameter_coordinates(args.hyperparams, embedding=args.projector)
    params_json["coordinates"] = parameter_space
    with open(JSON_PATH, "w") as f:
        json.dump(params_json, f, indent=4)

    # Subprocesses
    num_loops = len(parameter_space)

    projector = os.path.join(root, "src/projector.py")
    # Running Grid Search in Parallel
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for i, coord in enumerate(parameter_space):
        cmd = [
            "python",
            f"{projector}",
            f"-i {i}",
        ]
        subprocesses.append(cmd)

    # Running processes in Parallel
    # TODO: optimize based on max_workers
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses
        progress_bar = tqdm(total=num_loops, desc="Progress", unit="subprocess")
        for future in as_completed(futures):
            # Update the progress bar for each completed subprocess
            progress_bar.update(1)
    progress_bar.close()
