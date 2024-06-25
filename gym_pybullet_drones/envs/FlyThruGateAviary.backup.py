import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class FlyThruGateAviary(BaseRLAviary):
    """Single agent RL problem: fly through a gate with dynamic obstacle avoidance and precise control."""
    
    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results'
                 ):
        """Initialization of a single agent RL environment with advanced flight dynamics and environmental interaction."""
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=vision_attributes,
                         output_folder=output_folder
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
    
    def _computeDone(self):
        """Check if the episode is done based on the time elapsed."""
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    ################################################################################
    
    def _computeInfo(self):
        """Provide additional info useful for debugging or complex policy learning."""
        return {"answer": 42}  # Reference to the "Hitchhiker's Guide to the Galaxy"

    ################################################################################
    
    def _clipAndNormalizeState(self, state):
        """Normalize the drone's state to be within [-1,1] after clipping to operational limits."""
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1
        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        normalized_pos_xy= clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # Yaw angle normalized
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector are out of the clipping range.
        
        """
        if not np.all(clipped_pos_xy == state[0:2]):
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if clipped_pos_z != state[2]:
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not np.all(clipped_rp == state[7:9]):
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not np.all(clipped_vel_xy == state[10:12]):
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if clipped_vel_z != state[12]:
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

