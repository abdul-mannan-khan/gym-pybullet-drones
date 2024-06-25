import os
import numpy as np
import pybullet as p
import pkg_resources

import gymnasium
from gymnasium import spaces
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class FlyThruGateAviary(BaseRLAviary):
    """Single agent RL problem: fly through a gate with dynamic obstacle avoidance and precise control."""
    
    ################################################################################
    
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
                 target_pos=np.array([1,1,1])
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

    ################################################################################
    
    def _addObstacles(self):
        """Add obstacles to the environment, including a gate constructed from multiple elements."""
        super()._addObstacles()
        p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
                   [0, -1, .55],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        for i in range(10):
            p.loadURDF("cube_small.urdf",
                       [-.3, -1, .02 + i * 0.05],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [.3, -1, .02 + i * 0.05],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )

    ################################################################################
    
    def _computeReward(self):
        """Compute the reward based on the proximity to the gate and alignment through it."""
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter / self.PYB_FREQ) / self.EPISODE_LEN_SEC
        return -10 * np.linalg.norm(np.array([0, -2 * norm_ep_time, 0.75]) - state[0:3]) ** 2

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
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

    ################################################################################
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
gymnasium.register(
    id='flythrugate-aviary-v0',
    entry_point='flythrugate_aviary:FlyThruGateAviary',  # Adjust if placed in a different module
    max_episode_steps=1000,  # Adjust as needed
)