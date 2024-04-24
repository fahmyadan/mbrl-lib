import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import mbrl.util.math

from .model import Ensemble
from .util import EnsembleLinearLayer
from .gaussian_mlp import GaussianMLP
from .basic_ensemble import BasicEnsemble


class PlaNetEnsemble(BasicEnsemble):




    def __init__(self,
        ensemble_size: int,
        device: Union[str, torch.device],
        member_cfg: omegaconf.DictConfig,
        action_size: int, 
        propagation_method: Optional[str] = None,):


        super().__init__(ensemble_size, device, member_cfg, propagation_method)

        print('ensemble success')
