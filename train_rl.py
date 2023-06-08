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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="ppo", help="algorithm to use for training")
args = parser.parse_args()

# Load algorithm configuration from file
algorithm_config_path = os.path.join("configs", args.algorithm + ".yaml")
with open(algorithm_config_path, "r") as f:
    algorithm_config = yaml.safe_load(f)

# Load environment configuration from file
environment_config_path = os.path.join("configs", "env.yaml")
with open(environment_config_path, "r") as f:
    environment_config = yaml.safe_load(f)

# Initialize the environment
env = Environment(dataset="train.csv", **environment_config)

# Set up logging directory
logs_path = os.path.join("logs", args.algorithm, str(int(time.time())))
os.makedirs(logs_path, exist_ok=True)

# Wrap the environment with Monitor for logging and DummyVecEnv for normalization
env = Monitor(env, filename=logs_path)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Initialize the model based on the chosen algorithm
if args.algorithm == "ppo":
    model = PPO("MlpPolicy", env, **algorithm_config["model"])
if args.algorithm == "ddpg":
    model = DDPG("MlpPolicy", env, **algorithm_config["model"])

# Train the model
model.learn(**algorithm_config["learn"])

# Save the trained model and the environment
model.save(os.path.join(logs_path, "model"))
env.save(os.path.join(logs_path, "env.pkl"))

# Plot and save the training process
training_log = pd.read_csv(os.path.join(logs_path, "monitor.csv"), skiprows=1)
training_log["r"].plot()
plt.savefig(os.path.join(logs_path, "training_plot.png"))