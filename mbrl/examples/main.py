# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.env
import wandb
from callbacks import WandbCallback

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.overrides.get('logging', None) and cfg.overrides.logging.get('wandb', None):
        wandb_cfg = omegaconf.OmegaConf.to_container(cfg)
        wandb_run = wandb.init(project= cfg.overrides.logging.project_name, config=wandb_cfg, sync_tensorboard=True, monitor_gym=True)
        wanb_cbs = [WandbCallback('loss', wandb_run), WandbCallback('reward', wandb_run)]
    else :
        wanb_cbs = None
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg, callbacks=wanb_cbs)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":

        if not cfg.overrides.evaluate_trained:
            return planet.train(env, cfg, wandb=wanb_cbs)
        else: 
            model_path = cfg.overrides.logging.model_path
            return planet.evaluate_trained_model(model_path, env, cfg)


if __name__ == "__main__":
    run()
