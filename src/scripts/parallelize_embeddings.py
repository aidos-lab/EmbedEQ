"Project cleaned data using UMAP."

import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d  [%(levelname)-10s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "/src/")
    from loaders.factory import load_parameter_file, project_root_dir

    params = load_parameter_file()

    folder = project_root_dir() + f"/experiments/{params.run_name}/configs/"

    # Write Coordinates to JSON
    logging.info(f"Projectors: {params.embedding.model}")
    logging.info(f"Data sets: {params.data.dataset}")
    logging.info(f"Sample Sizes: {params.data.num_samples}")
    logging.info(f"Hyperparameters: {params.embedding.hyperparams}")

    # Subprocesses
    num_loops = len(os.listdir(folder))

    projector = os.path.join(root, "src/embed.py")
    # Running Grid Search in Parallel
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for i in range(num_loops):
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
