import itertools
from dataclasses import dataclass, field
from typing import Any

import models
from data import BaseDataConfig
from loaders.factory import LoadClass, load_parameter_file

#  ╭──────────────────────────────────────────────────────────╮
#  │Configs                                                   │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass(frozen=True)
class Config:
    meta: Any
    data: Any
    model: Any


@dataclass
class Meta:
    name: str
    id: int
    description: str
    project: str = "EmbedEQ"
    tags: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    config: Any


@dataclass
class ModelConfig:
    config: Any


#  ╭──────────────────────────────────────────────────────────╮
#  │Config Mappings                                           │
#  ╰──────────────────────────────────────────────────────────╯

configs = {
    "umap": "UMAPConfig",
    "isomap": "IsomapConfig",
    "phate": "PHATEConfig",
    "lle": "LLEConfig",
    "tsne": "TSNEConfig",
}
projectors = {
    "umap": "UMAPProjector",
    "isomap": "IsomapProjector",
    "phate": "PHATEProjector",
    "lle": "LLEProjector",
    "tsne": "TSNEProjector",
}
