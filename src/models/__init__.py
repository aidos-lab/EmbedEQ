######################################################################
# Silencing UMAP Warnings
import os
import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.simplefilter("ignore")

os.environ["KMP_WARNINGS"] = "off"


from .base import BaseConfig, BaseProjector
from .isomap import IsomapConfig, IsomapProjector
from .lle import LLEConfig, LLEProjector
from .phate import PHATEConfig, PHATEProjector
from .tsne import TSNEConfig, TSNEProjector
from .umap import UMAPConfig, UMAPProjector
