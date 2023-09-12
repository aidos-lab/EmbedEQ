import itertools

import models
from config import *
from loaders.factory import (
    LoadClass,
    create_experiment_folder,
    load_parameter_file,
    save_config,
)

#  ╭──────────────────────────────────────────────────────────╮
#  │Setup Experiments and Log Parameters                      │
#  ╰──────────────────────────────────────────────────────────╯


def configure_models():
    """Initilize models from `params.yaml`"""
    params = load_parameter_file().embedding

    ModelConfigs = []
    for model in params.model:
        class_name = configs[model]
        cfg = getattr(models, class_name)
        grid_search = list(
            itertools.product(
                *[params.hyperparams[attr] for attr in cfg.__annotations__]
            )
        )
        for parameter_combo in grid_search:
            cfg_params = dict(zip(cfg.__annotations__.keys(), parameter_combo))
            # Instantiate with necessary params
            cfg_params["name"] = model
            ModelConfigs.append(LoadClass.instantiate(cfg, cfg_params))

    return ModelConfigs


def configure_datasets():
    """Initilize datas from `params.yaml`"""
    params = load_parameter_file().data

    DataConfigs = []

    grid_search = list(itertools.product(params.dataset, params.num_samples))
    for dataset, n in grid_search:
        DataConfigs.append(
            BaseDataConfig(
                generator=dataset,
                num_samples=n,
                seed=params.seed,
            )
        )
    return DataConfigs


def generate_experiments():
    """
    This function generates model and dataset configurations, and saves them as `yaml` files.
    """
    params = load_parameter_file()
    folder = create_experiment_folder()

    models = configure_models()
    datasets = configure_datasets()

    experiments = list(itertools.product(models, datasets))

    for i, (model, data) in enumerate(experiments):
        meta = Meta(
            name=params.run_name,
            id=i,
            description=params.description,
        )
        config = Config(meta, data, model)
        save_config(config, folder, filename=f"config_{i}.yaml")

    # Copy Parameter Driver
    save_config(params, folder + "/..", f"{params.run_name}.yaml")


if __name__ == "__main__":
    generate_experiments()
