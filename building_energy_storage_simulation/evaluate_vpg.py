import time
import yaml
import gymnasium
import os
import pandas as pd
from matplotlib import pyplot as plt
import sys
import numpy as np

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
env = Environment(dataset="test.csv", **environment_config)



# Set up logging directory
logs_path = os.path.join("logs", 'vpg', str(int(time.time())))
os.makedirs(logs_path, exist_ok=True)

# Wrap the environment with Monitor for logging and DummyVecEnv for normalization
env = Monitor(env, filename=logs_path)
#env = DummyVecEnv([lambda: env])
#env = VecNormalize(env, norm_obs=True, norm_reward=True)
def load_agent(agent, filename):
    agent.policy_net.load_state_dict(torch.load(f'saved_agents/{filename}.pt'))
    agent.policy_net.eval()

def evaluate_agent(saved_agent_filename, saved_env_filename, num_episodes=10):
    loaded_agent = VPG(env)  # Create an instance of the VPG agent
    load_agent(loaded_agent, saved_agent_filename)  # Load the saved agent weights

    rewards = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        episode_rewards =[]
        while not done:
            action = loaded_agent.get_action(state)
            new_state, reward, done, _, info = env.step(action)
            total_reward += reward
            state = new_state
            episode_rewards.append(reward)
        rewards.append(episode_rewards)
        print(f"Episode: {episode} average reward: {np.mean(rewards[episode])}")

    episodes = range(num_episodes)
    avg_rewards = [np.mean(row) for row in rewards]

    plt.plot(episodes, avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Episode vs Average reward using {saved_agent_filename}')
    plt.savefig('test.png')
    plt.show()

evaluate_agent('trained_vpg', 'env.pkl', num_episodes=5)
