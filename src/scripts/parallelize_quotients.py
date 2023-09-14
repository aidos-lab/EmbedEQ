"Project cleaned data using UMAP."

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

n_cpus = int(os.cpu_count() / 2)


if __name__ == "__main__":
    # Load Params
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d  [%(levelname)-10s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "/src/")
    from loaders.factory import load_parameter_file

    params = load_parameter_file()

    # Subprocesses
    num_loops = (
        len(params.data.dataset)
        * len(params.embedding.model)
        * len(params.data.num_samples)
    )

    logging.info(f"Projector: {params.embedding.model}")
    logging.info(f"Data Sets: {params.data.dataset}")
    logging.info(f"Sample Sizes: {params.data.num_samples}")
    logging.info(f"Number of clustering models to generate: {num_loops}")

    clusterer = os.path.join(root, "src/quotient.py")
    subprocesses = []
    ### Generate Diagrams
    for alg in params.embedding.model:
        for dataset in params.data.dataset:
            for sample_size in params.data.num_samples:
                cmd = [
                    "python",
                    f"{clusterer}",
                    f"-d {dataset}",
                    f"-p {alg}",
                    f"-n {sample_size}",
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
