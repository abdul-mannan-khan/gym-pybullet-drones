import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

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
                 target_pos=np.array([4,4,4])
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = target_pos
        self.EPISODE_LEN_SEC = 8
        self.prev_distance_to_target = None

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
    
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        current_distance_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])

        norm_ep_time = (self.step_counter / self.PYB_FREQ) / self.EPISODE_LEN_SEC
        
        time_penalty = -1 * norm_ep_time
        distance_penalty = -10 * current_distance_to_target

        # Calculate distance change reward
        if self.prev_distance_to_target is not None:
            distance_change = self.prev_distance_to_target - current_distance_to_target
            if distance_change > 0:
                # Positive reward for moving closer to the target
                distance_change_reward = 20 * distance_change
            else:
                # Negative reward for moving further away from the target
                distance_change_reward = 20 * distance_change
        else:
            # If prev_distance_to_target is None, it's the first step
            distance_change_reward = 0

        # Calculate total reward
        total_reward = distance_penalty + time_penalty + distance_change_reward

        # Add a reward for reaching the target
        if current_distance_to_target < 0.1:
            total_reward += 350

        # Update the previous distance
        self.prev_distance_to_target = current_distance_to_target

        return total_reward



        #reward = -(distance_to_target**2)  # Use square of the distance
 
        # ret = max(0, 1500 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        # return ret
        # return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.05:
            print ("--------------------------------------------------")
            print ("             The drone reached goal.              ")
            print ("--------------------------------------------------")
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
        
        # Define the arena boundaries
        ARENA_SIZE_X = 15.0
        ARENA_SIZE_Y = 15.0
        ARENA_SIZE_Z = 15.0

        # Truncate if the drone is outside the defined arena boundaries
        if (abs(state[0]) > ARENA_SIZE_X or abs(state[1]) > ARENA_SIZE_Y or abs(state[2]) > ARENA_SIZE_Z
            or abs(state[7]) > .4 or abs(state[8]) > .4):  # Truncate when the drone is too tilted
            # print ("--------------------------------------------------")
            # print ("   The drone has gone out of working boundary.    ")
            # print ("--------------------------------------------------")
            return True
        
        # Truncate if the episode has timed out
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print ("--------------------------------------------------")
            print ("   Time is up. Too long to reach to the goal.     ")
            print ("--------------------------------------------------")
            return True
        
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
