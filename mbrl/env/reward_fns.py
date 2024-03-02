# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)

def highway_env(act: torch.Tensor, next_obs: torch.Tensor, collision_threshold=2.0) -> torch.Tensor:

    ego_obs = next_obs[:, :7]
    other_vehs = next_obs[:, 7:]
    other_vehs = other_vehs.view(other_vehs.shape[0], -1,ego_obs.shape[1])

    rel_x, rel_y = other_vehs[:, :, 1], other_vehs[:, :, 2]

    dist_squared = rel_x**2 + rel_y**2

    collision_mask = dist_squared < (collision_threshold**2)

    #TODO: Add path tracking

    return -1 * collision_mask.sum(dim=1).unsqueeze(-1)