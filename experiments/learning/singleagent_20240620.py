import os
import time
from datetime import datetime
from sys import platform
import subprocess
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary

from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary

import gym_pybullet_drones.utils.enums
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')#('one_d_rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False


from gym_pybullet_drones.utils.enums import ActionType


import shared_constants

EPISODE_REWARD_THRESHOLD = -0 # Upperbound: rewards are always negative, but non-zero
"""float: Reward threshold to halt the script."""
DEFAULT_ENV = 'hover'#'flythrugate'#
DEFAULT_ALGO = 'ppo'
DEFAULT_CPU = 1
DEFAULT_STEPS = int(1e7)
DEFAULT_OUTPUT_FOLDER = 'results'

def run(
    multiagent=DEFAULT_MA, 
    output_folder=DEFAULT_OUTPUT_FOLDER, 
    gui=DEFAULT_GUI, 
    plot=True, 
    colab=DEFAULT_COLAB, 
    record_video=DEFAULT_RECORD_VIDEO, 
    env=DEFAULT_ENV,
    algo=DEFAULT_ALGO,
    obs=DEFAULT_OBS,
    act=DEFAULT_ACT,
    cpu=DEFAULT_CPU,
    steps=DEFAULT_STEPS,
    local=True):
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
    if env == 'tune' and act != ActionType.TUN:
        print("\n\n\n[WARNING] TuneAviary is intended for use with ActionType.TUN\n\n\n")
    if act == ActionType.ONE_D_RPM or act == ActionType.ONE_D_DYN or act == ActionType.ONE_D_PID:
        print("\n\n\n[WARNING] Simplified 1D problem for debugging purposes\n\n\n")
    #### Errors ################################################
        if not env in ['takeoff', 'hover']: 
            print("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
            exit()
    # if act == ActionType.TUN and env != 'tune' :
    #     print("[ERROR] ActionType.TUN is only compatible with TuneAviary")
    #     exit()
    if algo in ['sac', 'td3', 'ddpg'] and cpu!=1: 
        print("[ERROR] The selected algorithm does not support multiple environments")
        exit()

    #### Uncomment to debug slurm scripts ######################
    env_name = env+"-aviary-v0"
    #sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs, act=act)
    sa_env_kwargs = dict(obs=obs, act=act)
    if not multiagent:
        if env_name == "takeoff-aviary-v0":
            train_env = make_vec_env(TakeoffAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=cpu,
                                    seed=0
                                    )
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
        if env_name == "tune-aviary-v0":
            train_env = make_vec_env(TuneAviary,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=cpu,
                                    seed=0
                                    )
        
        print("[INFO] Action space:", train_env.action_space)
        print("[INFO] Observation space:", train_env.observation_space)
        # check_env(train_env, warn=True, skip_render_check=True)
        # train_env = make_vec_env(HoverAviary,
        #                          env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
        #                          n_envs=1,
        #                          seed=0
        #                          )

        # Define policy keyword arguments based on the observation type
        if DEFAULT_OBS == ObservationType.KIN:
            policy_kwargs = {
                "activation_fn": torch.nn.ReLU,
                "net_arch": [512, 512, {"vf": [256, 128], "pi": [256, 128]}]  # Separate networks for actor and critic
            }
        else:
            policy_kwargs = {
                "activation_fn": torch.nn.ReLU,
                "net_arch": [512, 256, 128]  # Simplified architecture for visual observations
            }

        # Setup the environment with appropriate action and observation types
        env_config = {
            'obs': DEFAULT_OBS,
            'act': DEFAULT_ACT
        }
        
        train_env = make_vec_env(env_name, n_envs=1, env_kwargs=env_config)
        eval_env = make_vec_env(env_name, n_envs=1, env_kwargs=env_config)

        # Select the model based on the algorithm
        #algo = 'ppo'  # 'a2c', 'sac', 'td3', 'ddpg'
        # model_class = {
        #     'a2c': A2C,
        #     'ppo': PPO,
        #     'sac': SAC,
        #     'td3': TD3,
        #     'ddpg': DDPG
        # }.get(algo)

        #eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    model_classes = {
        'a2c': A2C,
        'ppo': PPO,
        'sac': SAC,
        'td3': TD3,
        'ddpg': DDPG
    }

    # model = PPO('MlpPolicy',
    #             train_env,
    #             verbose=1)

    # Select the correct policy class based on observation type
    policy_class = 'MlpPolicy' if DEFAULT_OBS == ObservationType.KIN else 'CnnPolicy'

    # Instantiate the model with dynamically selected class and policy
    model_class = model_classes[DEFAULT_ALGO]
    model = model_class(
                        policy_class,
                        train_env,
                        verbose=1,  
                        tensorboard_log=f"{filename}/tb/"
                        )

    target_reward = 474.15 if not multiagent else 949.5
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e7) if local else int(1e2),
                callback=eval_callback,
                log_interval=100)

    model.save(filename+'/final_model.zip')
    print(filename)

    if local:
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    # model = PPO.load(path)
    loaded_model = model_class.load(path)

    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                    num_drones=DEFAULT_AGENTS,
                                    obs=DEFAULT_OBS,
                                    act=DEFAULT_ACT,
                                    record=record_video)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab
                    )

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool, help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    
    parser.add_argument('--env',        default=DEFAULT_ENV,      type=str,             choices=['takeoff', 'hover', 'flythrugate', 'tune'], help='Task (default: hover)', metavar='')
    parser.add_argument('--algo',       default=DEFAULT_ALGO,        type=str,             choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'],        help='RL agent (default: ppo)', metavar='')
    parser.add_argument('--obs',        default=DEFAULT_OBS,        type=ObservationType,                                                      help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',        default=DEFAULT_ACT,  type=ActionType,                                                           help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu',        default=DEFAULT_CPU,          type=int,                                                                  help='Number of training environments (default: 1)', metavar='')        
    parser.add_argument('--steps',        default=DEFAULT_STEPS,          type=int,                                                                  help='Number of training time steps (default: 35000)', metavar='')        
    
    ARGS = parser.parse_args()
    
    ARGS = parser.parse_args()
    run(**vars(ARGS))
