import os
import pickle
import shutil
from dataclasses import fields

import numpy as np
import omegaconf
from dotenv import load_dotenv
from omegaconf import OmegaConf

creation_funcs = {}


class LoadClass:
    """Unpack an input dictionary to load a class"""

    classFieldCache = {}

    @classmethod
    def instantiate(cls, classToInstantiate, argDict):
        if classToInstantiate not in cls.classFieldCache:
            cls.classFieldCache[classToInstantiate] = {
                f.name for f in fields(classToInstantiate) if f.init
            }

        fieldSet = cls.classFieldCache[classToInstantiate]
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return classToInstantiate(**filteredArgDict)


def project_root_dir():
    load_dotenv()
    root = os.getenv("root")
    return root


def create_experiment_folder():
    name = load_parameter_file().run_name
    root = project_root_dir()
    path = root + f"/experiments/{name}/configs/"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def load_parameter_file():
    load_dotenv()
    YAML_PATH = os.getenv("params")
    params = OmegaConf.load(YAML_PATH)
    return params


def load_local_data(name):
    load_dotenv()
    file = os.getenv(name)
    return np.load(file, allow_pickle=True)


def save_config(cfg, folder, filename):
    path = os.path.join(folder, filename)
    c = OmegaConf.create(cfg)
    with open(path, "w") as f:
        OmegaConf.save(c, f)


def load_config(id, folder):
    path = os.path.join(folder, f"config_{id}.yaml")
    cfg = OmegaConf.load(path)
    return cfg


def load_configs_for_clustering(condition):
    configs = []
    root = project_root_dir()
    params = load_parameter_file()
    folder = root + f"/experiments/{params.run_name}/configs/"
    num_files = len(os.listdir(folder))
    for i in range(num_files):
        cfg = load_config(i, folder)
        try:
            if condition(cfg):
                configs.append(cfg)
        except omegaconf.errors.ConfigAttributeError as e:
            print("Config Attribute Error- Check sensitivity filter")
    return configs


def load_embedding(i):
    root = project_root_dir()
    params = load_parameter_file()
    folder = root + f"/experiments/{params.run_name}/configs/"
    cfg = load_config(i, folder)

    in_file = f"embedding_{i}.pkl"
    in_dir = os.path.join(
        root,
        "data/"
        + cfg.data.generator
        + "/"
        + cfg.meta.name
        + "/projections/"
        + cfg.model.name
        + "/",
    )
    in_file = os.path.join(in_dir, in_file)
    assert os.path.isfile(in_file), "Invalid Projection"

    with open(in_file, "rb") as f:
        D = pickle.load(f)
        X = D["projection"]
    return X


def load_diagrams(condition):
    root = project_root_dir()
    cfgs = load_configs_for_clustering(condition)
    keys = []
    diagrams = []
    for cfg in cfgs:
        in_dir = os.path.join(
            root,
            "data/"
            + cfg.data.generator
            + "/"
            + cfg.meta.name
            + "/diagrams/"
            + cfg.model.name
            + "/",
        )
        in_file = os.path.join(in_dir, f"diagram_{cfg.meta.id}.pkl")
        assert os.path.isfile(in_file), "Invalid Transform"

        with open(in_file, "rb") as f:
            data = pickle.load(f)
        keys.append(f"config_{cfg.meta.id}")
        diagrams.append(data["diagram"])

    return keys, diagrams


def load_distances(data, projector, metric, num_samples):
    root = project_root_dir()
    params = load_parameter_file()
    distances_in_file = os.path.join(
        root,
        "data/"
        + data
        + "/"
        + params.run_name
        + "/EQC/"
        + projector
        + "/distance_matrices/"
        + f"{metric}_{num_samples}_pairwise_distances.pkl",
    )
    with open(distances_in_file, "rb") as D:
        reference = pickle.load(D)
    return reference["keys"], reference["distances"]


def load_model(data, projector, metric, num_samples, linkage, dendrogram_cut):
    root = project_root_dir()
    params = load_parameter_file()
    model_in_file = os.path.join(
        root,
        "data/"
        + data
        + "/"
        + params.run_name
        + "/EQC/"
        + projector
        + "/models/"
        + f"embedding_{num_samples}_clustering_{metric}_{linkage}-linkage_{dendrogram_cut}.pkl",
    )

    with open(model_in_file, "rb") as M:
        model = pickle.load(M)["model"]
    return model
