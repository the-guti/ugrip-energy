import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.modules["gym"] = gym
import time
import yaml
import gym
import pandas as pd
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from building_energy_storage_simulation import Environment
from algorithms.A2C_continuous import ACAgent


def main():
    # Load environment configuration from file
    environment_config_path = os.path.join("configs", "env.yaml")
    with open(environment_config_path, "r") as f:
        environment_config = yaml.safe_load(f)

    # Initialize the environment
    env = Environment(dataset="train.csv", **environment_config)

    # Set up logging directory
    logs_path = os.path.join("logs", 'a2c', str(int(time.time())))
    os.makedirs(logs_path, exist_ok=True)

    # Wrap the environment with Monitor for logging and DummyVecEnv for normalization
    env = Monitor(env, filename=logs_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ACAgent(state_dim, action_dim)

    num_episodes = 100

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            agent.update(state, action, next_state, reward, done)
            state = next_state

        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    env.close()
    save_agent(agent, 'a2c_trained')


def save_agent(agent, filename):
    if not os.path.exists('saved_agents'):
        os.makedirs('saved_agents')
    torch.save(agent.actor.state_dict(), f'saved_agents/{filename}_actor.pt')
    torch.save(agent.critic.state_dict(), f'saved_agents/{filename}_critic.pt')


if __name__ == "__main__":
    main()