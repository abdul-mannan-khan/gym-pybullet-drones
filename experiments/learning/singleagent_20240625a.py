import os
import time
from datetime import datetime
from sys import platform
import argparse
import subprocess
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy, CnnPolicy as TD3CnnPolicy

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

import shared_constants

EPISODE_REWARD_THRESHOLD = -0  # Upperbound: rewards are always negative, but non-zero

DEFAULT_ENV = 'flythrugate'
DEFAULT_ALGO = 'ppo'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_CPU = 1
DEFAULT_STEPS = 35000
DEFAULT_OUTPUT_FOLDER = 'results'

def run(
    env=DEFAULT_ENV,
    algo=DEFAULT_ALGO,
    obs=DEFAULT_OBS,
    act=DEFAULT_ACT,
    cpu=DEFAULT_CPU,
    steps=DEFAULT_STEPS,
    output_folder=DEFAULT_OUTPUT_FOLDER
):

    #### Save directory ########################################
    filename = os.path.join(output_folder, 'save-'+env+'-'+algo+'-'+obs.value+'-'+act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    if (platform == "linux" or platform == "darwin") and ('GITHUB_ACTIONS' not in os.environ.keys()):
        git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
        with open(filename+'/git_commit.txt', 'w+') as f:
            f.write(str(git_commit))

    #### Warning ###############################################
    if act == ActionType.ONE_D_RPM or act == ActionType.ONE_D_DYN or act == ActionType.ONE_D_PID:
        print("\n\n\n[WARNING] Simplified 1D problem for debugging purposes\n\n\n")
        if not env in ['takeoff', 'hover']: 
            print("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
            exit()
    if algo in ['sac', 'td3', 'ddpg'] and cpu != 1: 
        print("[ERROR] The selected algorithm does not support multiple environments")
        exit()

    env_name = env+"-aviary-v0"
    sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs, act=act)
    
    if env_name == "hover-aviary-v0":
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=cpu,
                                 seed=0
                                 )
    if env_name == "flythrugate-aviary-v0":
        train_env = make_vec_env(FlyThruGateAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=cpu,
                                 seed=0
                                 )
    
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    #### On-policy algorithms ##################################
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[dict(pi=[512, 512], vf=[512, 512])]
                           )  # or None
    if algo == 'a2c':
        model = A2C(ActorCriticPolicy if obs == ObservationType.KIN else ActorCriticCnnPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )
    if algo == 'ppo':
        model = PPO(ActorCriticPolicy if obs == ObservationType.KIN else ActorCriticCnnPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )

    #### Off-policy algorithms #################################
    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[512, 512, 256, 128]
                            )  # or None
    if algo == 'sac':
        model = SAC(SACPolicy if obs == ObservationType.KIN else ActorCriticCnnPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )
    if algo == 'td3':
        model = TD3(TD3MlpPolicy if obs == ObservationType.KIN else TD3CnnPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )
    if algo == 'ddpg':
        model = DDPG(TD3MlpPolicy if obs == ObservationType.KIN else TD3CnnPolicy,
                     train_env,
                     policy_kwargs=offpolicy_kwargs,
                     tensorboard_log=filename+'/tb/',
                     verbose=1
                     )

    #### Create evaluation environment #########################
    if obs == ObservationType.KIN: 
        eval_env = gym.make(env_name,
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=obs,
                            act=act
                            )
    elif obs == ObservationType.RGB:
        if env_name == "takeoff-aviary-v0": 
            eval_env = make_vec_env(TakeoffAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        if env_name == "hover-aviary-v0": 
            eval_env = make_vec_env(HoverAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        if env_name == "flythrugate-aviary-v0": 
            eval_env = make_vec_env(FlyThruGateAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        if env_name == "tune-aviary-v0": 
            eval_env = make_vec_env(TuneAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=1,
                                    seed=0
                                    )
        eval_env = VecTransposeImage(eval_env)

    #### Train the model #######################################
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(2000/cpu),
                                 deterministic=True,
                                 render=False
                                 )
    model.learn(total_timesteps=steps,
                callback=eval_callback,
                log_interval=100,
                )

    #### Save the model ########################################
    model.save(filename+'/success_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--env', default=DEFAULT_ENV, type=str, choices=['takeoff', 'hover', 'flythrugate', 'tune'], help='Task (default: hover)', metavar='')
    parser.add_argument('--algo', default=DEFAULT_ALGO, type=str, choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'], help='RL agent (default: ppo)', metavar='')
    parser.add_argument('--obs', default=DEFAULT_OBS, type=ObservationType, help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType, help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu', default=DEFAULT_CPU, type=int, help='Number of training environments (default: 1)', metavar='')        
    parser.add_argument('--steps', default=DEFAULT_STEPS, type=int, help='Number of training time steps (default: 35000)', metavar='')        
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))