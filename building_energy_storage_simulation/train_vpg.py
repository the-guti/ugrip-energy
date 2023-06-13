import time
import yaml
import gymnasium
import os
import pandas as pd
from matplotlib import pyplot as plt
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DDPG, A2C, SAC
from building_energy_storage_simulation import Environment
from building_energy_storage_simulation.VPG import VPG
import pickle
import torch

# Load environment configuration from file
environment_config_path = os.path.join("configs", "env.yaml")
with open(environment_config_path, "r") as f:
    environment_config = yaml.safe_load(f)

# Initialize the environment
env = Environment(dataset="train.csv", **environment_config)



# Set up logging directory
logs_path = os.path.join("logs", 'vpg', str(int(time.time())))
os.makedirs(logs_path, exist_ok=True)

# Wrap the environment with Monitor for logging and DummyVecEnv for normalization
env = Monitor(env, filename=logs_path)
#env = DummyVecEnv([lambda: env])
#env = VecNormalize(env, norm_obs=True, norm_reward=True)

def save_agent(agent, filename):
    if not os.path.exists('saved_agents'):
        os.makedirs('saved_agents')
    torch.save(agent.policy_net.state_dict(), f'saved_agents/{filename}.pt')

def trainAndPlot(env, agent, save_agent_filename=None, save_env_filename=None):
    rewards, averageRewardsPerEpisode = agent.train()
    if save_agent_filename:
        save_agent(agent, save_agent_filename)
        print('saved agent')
    # if save_env_filename:
    #     save_env(env, save_env_filename)
    #     print('saved env')

    episodes = [row[0] for row in averageRewardsPerEpisode]
    avg_rewards = [row[1] for row in averageRewardsPerEpisode]

    plt.plot(episodes, avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.title('Episode vs Average reward using our VPG agent in our env')
    plt.savefig('train.png')
    plt.show()

agent = VPG(env)
trainAndPlot(env, agent, 'trained_vpg', 'env' )
