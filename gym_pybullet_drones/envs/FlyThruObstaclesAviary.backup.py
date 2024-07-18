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
                 target_pos=np.array([4,4,4])
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
    # def _addObstacles(self):
    #     """Add a single red cylinder to the environment at a fixed position."""
    #     super()._addObstacles()

    #     # List of fixed positions for the red cylinders
    #     fixed_positions = [
    #         (2, 2, 5),
    #         (-2, -2, 5),
    #         (3, 0, 5),
    #         (-3, 0, 5),
    #         (0, 3, 5),
    #         (-2, 3, 5),
    #         (-2, -3, 5),
    #         (5, 4, 5)
    #     ]

    #     urdf_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf')
    #     for pos in fixed_positions:
    #         p.loadURDF(urdf_path,
    #                    pos,
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
            
    # def _addObstacles(self):
    #     """Add a single red cylinder to the environment at a fixed position."""
    #     super()._addObstacles()

    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [2, 2, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
        
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [-2, -2, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
        
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [3, 0, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [-3, 0, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [0, 3, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [-2, 3, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [-2, -3, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf'),
    #                [5, 4, 5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
        
        # urdf_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/red_cylinder.urdf')
        # fixed_position = (2, 2, 5)  # Specified fixed position for the red cylinder

        # if os.path.exists(urdf_path):
        #     p.loadURDF(urdf_path, basePosition=fixed_position, useFixedBase=False)
        # else:
        #     print(f"File not found: {urdf_path}")
    
    def _addObstacles(self):
        """Add obstacles to the environment, including multiple cylinders of different colors at fixed positions."""
        super()._addObstacles()
        base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
        cylinder_colors = ['red', 'orange', 'green']
        cylinders = [os.path.join(base_path, f"{color}_cylinder.urdf") for color in cylinder_colors for _ in range(3)]

        # Fixed positions
        fixed_positions = [
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

        for urdf, pos in zip(cylinders, fixed_positions):
            if os.path.exists(urdf):
                p.loadURDF(urdf,
                       pos,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
                # print(f"File loaded: {urdf}")
            else:
                print(f"File not found: {urdf}")

    # --------------------------------------------------------------
    ##        Code for placing Obstacles at Random Places
    #---------------------------------------------------------------
    # def _addObstacles(self):
    #     """Add obstacles to the environment, including multiple cylinders of different colors."""
    #     super()._addObstacles()
    #     num_cylinders_per_color = 3  # Three cylinders per color
    #     min_distance = 5.0  # Minimum distance between any two cylinders
    #     exclusion_zone = 1.0  # No cylinder within this distance from the target position
    #     base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
    #     cylinder_colors = ['red', 'orange', 'green']
    #     cylinders = [os.path.join(base_path, f"{color}_cylinder.urdf") for color in cylinder_colors for _ in range(num_cylinders_per_color)]

    #     positions = []
    #     while len(positions) < len(cylinders):
    #         new_pos = [random.uniform(-15, 15), random.uniform(-15, 15), 5.0]  # Random positions
    #         if self.is_valid_position(new_pos, positions, min_distance, exclusion_zone):
    #             positions.append(new_pos)

    #     for urdf, pos in zip(cylinders, positions):
    #         if os.path.exists(urdf):
    #             p.loadURDF(urdf, 
    #                       basePosition=pos, 
    #                       p.getQuaternionFromEuler([0, 0, 0]), 
    #                       physicsClientId=self.CLIENT)
    #         else:
    #             print(f"File not found: {urdf}")
    
    # def is_valid_position(self, new_pos, existing_positions, min_distance, exclusion_zone):
    #     """Check if the new position is at least min_distance away from all existing positions and not within the exclusion zone of the target."""
    #     x_new, y_new, _ = new_pos  # Ignore the z-coordinate
    #     # Check distance from target position to ensure it's not within the exclusion zone
    #     if np.linalg.norm(np.array(new_pos[:2]) - self.TARGET_POS[:2]) < exclusion_zone:
    #         return False
    #     for pos in existing_positions:
    #         x_existing, y_existing, _ = pos
    #         distance = math.sqrt((x_new - x_existing)**2 + (y_new - y_existing)**2)
    #         if distance < min_distance:
    #             return False
    #     return True

    def _computeReward(self):
        """Compute the reward based on the proximity to the target position."""
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter / self.PYB_FREQ) / self.EPISODE_LEN_SEC
        #time_reward = -10 * np.linalg.norm(np.array([0, -2 * norm_ep_time, 0.75]) - state[0:3]) ** 2
        time_penalty = -10 * norm_ep_time
        distance_penalty = -np.linalg.norm(self.TARGET_POS - state[0:3])
        total_reward = distance_penalty + time_penalty
        # return -np.linalg.norm(self.TARGET_POS - state[0:3])
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
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
# Register the environment with gymnasium
# Uncomment below to register the environment
# gymnasium.register(
#     id='flythruobstacles-aviary-v0',
#     entry_point='thismodule:FlyThruObstaclesAviary',  # Adjust the module path as necessary
#     max_episode_steps=1000,
# )
