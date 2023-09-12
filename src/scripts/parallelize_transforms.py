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
    from loaders.factory import project_root_dir

    YAML_PATH = os.getenv("params")
    assert os.path.isfile(YAML_PATH), "Please configure .env to point to params.yaml"
    with open(YAML_PATH, "r") as f:
        params = OmegaConf.load(YAML_PATH)

    folder = project_root_dir() + f"/experiments/{params.run_name}/configs/"
    num_loops = len(os.listdir(folder))

    # Write Coordinates to JSON
    logging.info(f"Data Sets: {params.data.dataset}")
    logging.info(
        f"Computing Persistent Homology with `{params.topology.filtration}` Filtration"
    )
    logging.info(f"Maximum Homology Dim: {params.topology.homology_max_dim}")
    logging.info(f"Number of diagrams to generate: {num_loops}")

    # Subprocesses
    transformer = os.path.join(root, "src/transform.py")
    # Running Grid Search in Parallel
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for i in range(num_loops):
        cmd = [
            "python",
            f"{transformer}",
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
