import os
import numpy as np
import pybullet as p
import random
import math
import pkg_resources
import gymnasium
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class FlyThruObstaclesAviary(BaseRLAviary):
    """Single agent RL environment: fly through obstacles with varying colors and dynamic placement."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 target_pos=np.array([4, 4, 4])
                 ):
        """Initialization of a single agent RL environment with advanced flight dynamics and environmental interaction."""
        self.TARGET_POS = target_pos
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
    def _addObstacles(self):
        """Add obstacles to the environment, including multiple cylinders of different colors at fixed positions."""
        super()._addObstacles()
        base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
        cylinder_colors = ['red', 'orange', 'green']
        cylinders = [os.path.join(base_path, f"{color}_cylinder.urdf") for color in cylinder_colors for _ in range(3)]

        # Fixed positions
        self.fixed_positions = [
            (2, 2, 5),
            (-2, -2, 5),
            (3, 0, 5),
            (-3, 0, 5),
            (0, 3, 5),
            (0, -3, 5),
            (-2, 3, 5),
            (-2, -3, 5),
            (5, 4, 5)
        ]

        for urdf, pos in zip(cylinders, self.fixed_positions):
            if os.path.exists(urdf):
                p.loadURDF(urdf,
                           pos,
                           p.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=self.CLIENT
                           )
            else:
                print(f"File not found: {urdf}")

    def _computeReward(self):
        """Compute the reward based on the proximity to the target position and distance from obstacles."""
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter / self.PYB_FREQ) / self.EPISODE_LEN_SEC
        
        time_penalty = -10 * norm_ep_time
        distance_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])
        distance_penalty = -distance_to_target

        # Calculate distances to all obstacles
        distances_to_obstacles = [np.linalg.norm(np.array(pos[:2]) - state[:2]) for pos in self.fixed_positions]
        min_distance_to_obstacle = min(distances_to_obstacles) if distances_to_obstacles else float('inf')
        obstacle_penalty = -1 / min_distance_to_obstacle if min_distance_to_obstacle > 0 else -float('inf')
        
        # Calculate reward
        total_reward = distance_penalty + time_penalty + obstacle_penalty
        # Add a reward for reaching the target
        if distance_to_target < 0.1:
            total_reward += 100
        # Add a penalty for hitting an obstacle
        if min_distance_to_obstacle < 0.05:
            total_reward -= 50

        return total_reward

    def _computeTerminated(self):
        """Check if the current episode is done."""
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.
        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0  # Truncate when the drone is too far away
                or abs(state[7]) > .4 or abs(state[8]) > .4  # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

# Register the environment with gymnasium
# Uncomment below to register the environment
# gymnasium.register(
#     id='flythruobstacles-aviary-v0',
#     entry_point='thismodule:FlyThruObstaclesAviary',  # Adjust the module path as necessary
#     max_episode_steps=1000,
# )
