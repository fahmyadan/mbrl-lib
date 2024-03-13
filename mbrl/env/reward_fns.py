# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns
from highway_env.road.lane import StraightLane, CircularLane
import numpy as np
device ='cuda' if torch.cuda.is_available() else 'cpu'
 
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

def highway_env(act: torch.Tensor, next_obs: torch.Tensor, veh_obj= None, collision_threshold=2.0,) -> torch.Tensor:

    lane_width = 4

    ego_obs = next_obs[:, :7]
    other_vehs = next_obs[:, 7:]
    other_vehs = other_vehs.view(other_vehs.shape[0], -1,ego_obs.shape[1])

    rel_x, rel_y = other_vehs[:, :, 1], other_vehs[:, :, 2]

    dist_squared = rel_x**2 + rel_y**2

    collision_mask = dist_squared < (collision_threshold**2)

    #TODO: Add path tracking
    veh_obj = veh_obj[0]
    lanes_ahead = [veh_obj.road.network.get_lane(r) for r in veh_obj.route] 
    traj = torch.tensor(global_path(lanes_ahead, veh_obj)).to(device)
    ego_next_position = ego_obs[:, 1:3]

    batch, n = ego_next_position.shape
    next_position = ego_next_position.reshape(batch, 1, n)
    distance = torch.norm(next_position - traj, dim= 2)

    min_distance = torch.min(distance, dim = 1)
    sq_error = min_distance.values **2 
    sq_thld = sq_error < lane_width /2

    reward = -1 * collision_mask.sum(dim=1) + 1 * sq_thld

    return reward.unsqueeze(-1)

def global_path(lanes_to_go, veh_obj):

    trajectory = []

    for lane in lanes_to_go: 
        if isinstance(lane, CircularLane):

            arc_points = lane.get_arc_points()
            nodeless_arc_points = arc_points[1:-1]
            trajectory.append(nodeless_arc_points)
        elif isinstance(lane, StraightLane): 
            startx, starty = lane.start[0], lane.start[1]
            endx, endy = lane.end[0], lane.end[1]
            x_n = np.linspace(startx, endx, 40)
            y_n = np.linspace(starty, endy, 40)
            new = np.column_stack((x_n, y_n))
            nodeless_new = new[1:-1]
            trajectory.append(nodeless_new)

    trajectory = np.concatenate(trajectory)
    closest_pos = trajectory[0]
    for idx, pos in enumerate(trajectory):

        if np.linalg.norm(veh_obj.position - pos) <= np.linalg.norm(veh_obj.position - closest_pos):
            closest_pos = pos
            closest_idx = idx

    return trajectory[closest_idx:]