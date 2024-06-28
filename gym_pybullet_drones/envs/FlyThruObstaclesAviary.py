import os
import numpy as np
import pybullet as p
import random
import math

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
                 target_pos=np.array([4,4,1])
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
        """Add obstacles to the environment, including multiple cylinders of different colors."""
        super()._addObstacles()
        num_cylinders_per_color = 3  # Three cylinders per color
        min_distance = 5.0  # Minimum distance between any two cylinders
        # Dynamic path based on script location
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
        cylinder_colors = ['red', 'orange', 'green']
        cylinders = [os.path.join(base_path, f"{color}_cylinder.urdf") for color in cylinder_colors for _ in range(num_cylinders_per_color)]

        # Generate random positions for each cylinder, ensuring they don't overlap
        positions = []
        while len(positions) < len(cylinders):
            new_pos = [random.uniform(-15, 15), random.uniform(-15, 15), 5.0]  # Random positions
            if self.is_valid_position(new_pos, positions, min_distance):
                positions.append(new_pos)

        # Load each cylinder at a random position
        for urdf, pos in zip(cylinders, positions):
            if os.path.exists(urdf):
                p.loadURDF(urdf, basePosition=pos, useFixedBase=False)
            else:
                print(f"File not found: {urdf}")

    def is_valid_position(self, new_pos, existing_positions, min_distance):
        """Check if the new position is at least min_distance away from all existing positions."""
        x_new, y_new, _ = new_pos  # Ignore the z-coordinate
        for pos in existing_positions:
            x_existing, y_existing, _ = pos
            distance = math.sqrt((x_new - x_existing)**2 + (y_new - y_existing)**2)
            if distance < min_distance:
                return False
        return True

    def _computeReward(self):
        """Compute the reward based on the proximity to the target position."""
        state = self._getDroneStateVector(0)
        return -np.linalg.norm(self.TARGET_POS - state[0:3])

    def _computeTerminated(self):
        """Check if the current episode is done."""
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1

    def _computeTruncated(self):
        """Check if the episode should be truncated."""
        state = self._getDroneStateVector(0)
        return (self.step_counter/self.PYB_FREQ) > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
# Register the environment with gymnasium
# Uncomment below to register the environment
# gymnasium.register(
#     id='flythruobstacles-aviary-v0',
#     entry_point='thismodule:FlyThruObstaclesAviary',  # Adjust the module path as necessary
#     max_episode_steps=1000,
# )
