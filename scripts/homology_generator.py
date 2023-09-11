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
        "-n",
        "--num_samples",
        default=params_json["num_samples"],
        type=int,
        help="Set number of samples in data set",
    )
    parser.add_argument(
        "-p",
        "--projector",
        type=str,
        default=params_json["projector"],
        help="Set to the name of projector for dimensionality reduction. ",
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
    num_loops = len(params_json["coordinates"]) * len(args.data) * len(
        args.num_samples
    ) + len(
        args.num_samples
    )  # original spapce
    homology = os.path.join(root, "src/homology.py")
    subprocesses = []
    ### Generate Diagrams
    for alg in args.projector:
        logging.info(f"Projector: {alg}")
        logging.info(f"Computing Persistent Homology with `{args.filter}` Filtration")
        logging.info(f"Maximum Homology Dim: {args.homology_max_dim}")
        logging.info(f"Data Sets: {args.data}")
        logging.info(f"Sample Sizes: {args.num_samples}")
        logging.info(f"Number of diagrams to generate: {num_loops}")
        for dataset in args.data:
            for sample_size in args.num_samples:
                original_space_cmd = [
                    "python",
                    f"{homology}",
                    f"-d {dataset}",
                    f"-n {sample_size}",
                    f"-p {alg}",
                    f"-i {-1}",
                    f"-M {args.homology_max_dim}",
                ]
                subprocesses.append(original_space_cmd)
                for i in range(len(params_json["coordinates"])):
                    cmd = [
                        "python",
                        f"{homology}",
                        f"-d {dataset}",
                        f"-n {sample_size}",
                        f"-p {alg}",
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
        print()
