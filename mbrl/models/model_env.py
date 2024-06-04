# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from highway_env import utils
import mbrl.types
from mbrl.env.HighwayEnv.highway_env.envs.intersection_env import IntersectionEnv

from . import Model
import logging

logger = logging.getLogger(__name__)


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]
    LENGTH = 5.0

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True
        self.obs_process_fn = obs_process_fn

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        if isinstance(self.dynamics_model, mbrl.models.OneDTransitionRewardModel):
            assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        with torch.no_grad():
            model_state = self.dynamics_model.reset(
                initial_obs_batch.astype(np.float32), rng=self._rng
            )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
            ) = self.dynamics_model.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
            )
            if not isinstance(self.env.unwrapped, IntersectionEnv):
                rewards = (
                    pred_rewards
                    if self.reward_fn is None
                    else self.reward_fn(actions, next_observs)
                )
            else:
                rewards = (
                    pred_rewards
                    if self.reward_fn is None
                    else self.reward_fn(actions, next_observs, self.env.unwrapped.controlled_vehicles)
                )

            if isinstance(self.env.unwrapped, IntersectionEnv):
                # dones = self.termination_fn(actions, next_observs, self.env)
                dones = self.termination_fn(actions, next_observs)
            else:
                dones = self.termination_fn(actions, next_observs)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )
            next_observations = self.dynamics_model._forward_vec_decoder(model_state['latent'], model_state['belief'])
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state, next_observations

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            #TODO: FIX MPPI ACTION Evaluation
            global tiling_shape
            if not isinstance(self.observation_space, gym.spaces.Tuple):
                assert initial_state.ndim in (1,2, 3)

                tiling_shape = (num_particles * population_size,) + tuple(
                    [1] * initial_state.ndim
                )
                initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            else: 
                # Second element of initial state is the kinematic obs 
                inital_mppi_pose = initial_state[1][0,1:4]   #x, y, theta
                initial_mppi_speed = np.linalg.norm(initial_state[1][0, 4:6])
                initial_mppi_obs = np.concatenate([inital_mppi_pose, initial_mppi_speed.reshape(1,)])
                tiling_shape =  (num_particles * population_size,) + tuple(
                    [1] * initial_mppi_obs.ndim
                )
                initial_obs_batch = np.tile(initial_mppi_obs, tiling_shape).astype(np.float32)


            if self.obs_process_fn:
                initial_obs_batch = self.obs_process_fn(initial_obs_batch)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            longitudinal_actions = action_sequences[:,:,0]
            steering_popn= []
            for time_step in range(horizon):
                steering_actions = self.steering_controller(initial_obs_batch, action_sequences)
                if any(torch.isnan(steering_actions)):
                    steering_actions = self._parse_nans(steering_actions, action_sequences[:, time_step, 1] )
                steering_popn.append(steering_actions)
                long_actions_for_step = longitudinal_actions[:,time_step].view(-1,1)
                action_for_step = torch.concatenate([long_actions_for_step, steering_actions], dim=-1)
                # action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state, next_observations  = self.step(
                    action_batch, model_state, sample=True
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards
                initial_obs_batch = next_observations[1]
            steering_tensor = torch.stack(steering_popn, dim=1)
            hybrid_action_seq = torch.concatenate([longitudinal_actions.unsqueeze(-1), steering_tensor], dim=-1)
            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1), hybrid_action_seq

    def steering_controller(self, obs_batch, actions, scale=100):

        #Get positions @ time t -> (xt, yt) for K samples 
        all_positions, all_speed, all_heading, all_closest_lane_idx = [], [], [], []
        steering_angles = []
        

        if isinstance(obs_batch, torch.Tensor):
            K = obs_batch.shape[0]
            np_obs = obs_batch.cpu().numpy()
            pose, velocity = np_obs[:, :3], np_obs[:,3:]
            speed = np.linalg.norm(velocity, axis=1).reshape(K, 1)
            obs_batch = np.concatenate([pose, speed], axis=-1)
            

        for idx, obs in enumerate(obs_batch):
            position_t = obs[:2] * scale
            all_positions.append(position_t)
            speed = obs[-1]
            heading = obs[2]
            closest_lane_idx = self.env.road.network.get_closest_lane_index(position_t)
            all_closest_lane_idx.append(closest_lane_idx)


        # Get current lane at @ time t and check if on the correct route -> Ln
            if closest_lane_idx in self.env.controlled_vehicles[0].route:
                current_lane = self.env.road.network.get_lane(closest_lane_idx)

                # Check if (xt, yt) is @ end of lane -> Lt (target lane)
                if self.env.road.network.get_lane(closest_lane_idx).after_end(position_t):
                    target_lane_index = self.env.road.network.next_lane(
                        closest_lane_idx,
                        route=self.env.controlled_vehicles[0].route,
                        position=position_t,
                        np_random=self.env.road.np_random,
                    )
                else:
                    target_lane_index = closest_lane_idx 
                
                steering = self.proportional_steering(target_lane_index, position_t, speed, heading)
                steering_angles.append(steering) 
            else: 
                # return some random steering action 
                target_lane_index = None 
                rand_steer = self.env.action_space.sample()[1]

                # logger.warning('Position is off route! Doing Random Steering action')
                steering_angles.append(float('nan'))

        # Implement steering_control logic 

            # if not target_lane_index:
            #     rand_steer = self.env.action_space.sample()[1]

            #     logger.warning('Position is off route! Doing Random Steering action')
            #     steering_angles.append((None, rand_steer))
            # else: 

            #     steering = self.proportional_steering(target_lane_index, position_t, speed, heading)

            # if steering:
            #     # steering_samples = np.tile([steering], tiling_shape)
            #     steering_angles.append(steering) 
        
        return torch.tensor(steering_angles, device=self.device).view(-1,1)

    



    def proportional_steering (self, target_lane_index, position, speed, heading) -> float:
            """
            Steer the vehicle to follow the center of an given lane.

            1. Lateral position is controlled by a proportional controller yielding a lateral speed command
            2. Lateral speed command is converted to a heading reference
            3. Heading is controlled by a proportional controller yielding a heading rate command
            4. Heading rate command is converted to a steering angle

            :param target_lane_index: index of the lane to follow
            :return: a steering wheel angle command [rad]
            """
            target_lane = self.env.road.network.get_lane(target_lane_index)
            lane_coords = target_lane.local_coordinates(position)
            lane_next_coords = lane_coords[0] + speed * self.TAU_PURSUIT
            lane_future_heading = target_lane.heading_at(lane_next_coords)

            # Lateral position control
            lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
            # Lateral speed to heading
            heading_command = np.arcsin(
                np.clip(lateral_speed_command / utils.not_zero(speed), -1, 1)
            )
            heading_ref = lane_future_heading + np.clip(
                heading_command, -np.pi / 4, np.pi / 4
            )
            # Heading control
            heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
                heading_ref - heading
            )
            # Heading rate to steering angle
            slip_angle = np.arcsin(
                np.clip(
                    self.LENGTH / 2 / utils.not_zero(speed) * heading_rate_command,
                    -1,
                    1,
                )
            )
            steering_angle = np.arctan(2 * np.tan(slip_angle))
            steering_angle = np.clip(
                steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
            )
            return float(steering_angle)

    def _parse_nans(self, steering, original_actions):

        return torch.where(torch.isnan(steering), original_actions.view(-1,1), steering)