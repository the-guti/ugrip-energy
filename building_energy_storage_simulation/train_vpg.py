import time
import yaml
import gymnasium
import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DDPG, A2C, SAC
from building_energy_storage_simulation import Environment
from building_energy_storage_simulation.VPG import VPG


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


def trainAndPlot(env, agent):
  returns, data = agent.train()
  episodes = [row[0] for row in data]
  returns = [row[1] for row in data]

  plt.plot(episodes, returns)
  plt.xlabel('Episode')
  plt.ylabel('Average Return')
  plt.title(f'Episode vs Average Return using our VPG agent in our env')
  plt.savefig(os.path.join(logs_path, "training_plot_vpg.png"))
  plt.show()

agent = VPG(env)
trainAndPlot(env, agent )
