"""Test script for single agent problems using Stable Baselines3 models.

To run this script from a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

Example
-------
Run with specific experiment folder to evaluate and test a trained model.
"""

import os
import time
import argparse
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import shared_constants

DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = 'results'

def run(exp, gui=DEFAULT_GUI, 
        plot=DEFAULT_PLOT, 
        output_folder=DEFAULT_OUTPUT_FOLDER):
    algo = exp.split("-")[2]
    model_path = os.path.join(exp, 'success_model.zip' if os.path.exists(os.path.join(exp, 'success_model.zip')) else 'best_model.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR]: Model not found under the specified path {model_path}")

    model_class = {'a2c': A2C, 'ppo': PPO, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG}[algo.lower()]
    model = model_class.load(model_path)

    env_name = exp.split("-")[1] + "-aviary-v0"
    obs_type = ObservationType.KIN if 'kin' in exp else ObservationType.RGB
    act_type = ActionType[exp.split("-")[4].upper()]

    eval_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs_type, act=act_type)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"\nMean reward: {mean_reward} ± {std_reward}\n")

    if gui or plot:
        test_env = gym.make(env_name, gui=gui, record=False, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs_type, act=act_type)
        obs = test_env.reset()
        start_time = time.time()
        for _ in range(360):  # 6 seconds, 60Hz
            if isinstance(obs, tuple):  # Handling Gym environment
                obs = obs[0]  # Only take the observation part if it's a tuple
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info, *_ = test_env.step(action)  # Adjusted to handle additional return values
            test_env.render()
            if done:
                break
            time.sleep(1 / test_env.PYB_FREQ)  # Keep the frequency consistent
        test_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for evaluating single agent models.")
    parser.add_argument('--exp', required=True, type=str, help='Experiment directory')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use GUI for rendering')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Plot the results')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Output folder for logs and plots')
    args = parser.parse_args()

    run(**vars(args))