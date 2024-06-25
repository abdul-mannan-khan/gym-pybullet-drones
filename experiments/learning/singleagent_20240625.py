"""Learning script for single agent problems with Stable Baselines3.

To run the script from the terminal:

    $ python singleagent.py --env <env> --algo <alg> --obs <ObservationType> --act <ActionType> --cpu <cpu_num>

Use TensorBoard to view training results:

    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/
    Access at http://localhost:6006/
"""

import os
import argparse
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

#from gym_pybullet_drones.envs.single_agent_rl import TakeoffAviary, HoverAviary, FlyThruGateAviary, TuneAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary

from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_ENV = 'flythrugate'
#DEFAULT_ENV = 'hover'
DEFAULT_ALGO = 'sac'
DEFAULT_OBS = ObservationType.KIN
DEFAULT_ACT = ActionType.ONE_D_RPM
DEFAULT_CPU = 1
DEFAULT_STEPS = 50000
DEFAULT_OUTPUT_FOLDER = 'results'

def run(env, algo, obs, act, cpu, steps, output_folder):
    output_path = os.path.join(output_folder, f'save-{env}-{algo}-{obs.value}-{act.value}-{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_path, exist_ok=True)

    env_class = globals()[f"{env.capitalize()}Aviary"]
    env = make_vec_env(env_class, n_envs=cpu, seed=0, env_kwargs={'obs': obs, 'act': act})

    model_class = {'a2c': A2C, 'ppo': PPO, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG}[algo]
    model = model_class("MlpPolicy", env, verbose=1, tensorboard_log=f"{output_path}/tb/")
    eval_env = make_vec_env(env_class, n_envs=1, env_kwargs={'obs': obs, 'act': act})

    callback = EvalCallback(eval_env, best_model_save_path=output_path, log_path=output_path, eval_freq=500)
    model.learn(total_timesteps=steps, callback=callback)
    model.save(os.path.join(output_path, 'final_model.zip'))

    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single agent using Stable Baselines3.")
    parser.add_argument('--env', type=str, default=DEFAULT_ENV, choices=['takeoff', 'hover', 'flythrugate', 'tune'], help="Environment ID")
    parser.add_argument('--algo', type=str, default=DEFAULT_ALGO, choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'], help="Algorithm")
    parser.add_argument('--obs', type=ObservationType, default=DEFAULT_OBS, help="Type of observation")
    parser.add_argument('--act', type=ActionType, default=DEFAULT_ACT, help="Type of action")
    parser.add_argument('--cpu', type=int, default=DEFAULT_CPU, help="Number of CPUs")
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help="Training steps")
    parser.add_argument('--output_folder', type=str, default=DEFAULT_OUTPUT_FOLDER, help="Output folder path")
    args = parser.parse_args()

    run(**vars(args))
