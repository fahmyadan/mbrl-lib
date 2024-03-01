# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch

# TODO remove act from all of these, it's not needed


def highway_env(obs: torch.Tensor):

    """
    args: obs 

    return: batch, flat_dims 
    """

    if obs.ndim > 2:
        return obs.reshape(*obs.shape[:-2], -1)

    return obs
    
     
