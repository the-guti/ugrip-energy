import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
import torch.optim as optim
from torch import softmax
from collections import deque
import yaml
import time
import wandb
from utils import *
from building_energy_storage_simulation.environment import Environment


class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim, linspace):
        super(NNModel, self).__init__()
        self.linspace = linspace
        self.fc1_critic = nn.Linear(24, 128)
        self.fc2_critic = nn.Linear(128, 1)
        self.fc1_actor = nn.Linear(24, 512)
        self.fc2_actor = nn.Linear(512, 50)

    def forward_critic(self, x):
        x = torch.relu(self.fc1_critic(x))
        return self.fc2_critic(x)

    def forward_actor(self, x):
        x = torch.relu(self.fc1_actor(x))
        return softmax(self.fc2_actor(x), dim=1)

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            action = Categorical(output).sample().item()
            return torch.tensor([[action]])

    def act(self, state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = self.forward_critic(state)
        probs = self.forward_actor(state)
        m = Categorical(probs)
        categorical_sampling = m.sample()
        action_indx = categorical_sampling.item()
        action = self.linspace[action_indx]
        log_probs = m.log_prob(categorical_sampling)
        return value, action, log_probs


class A2C:
    def __init__(self, dummy, env, config):
        self.environment = env
        self.gamma = 0.95
        self.learning_rate = 0.0003
        self.linspace = np.linspace(-0.99, 0.99, 50)
        torch.manual_seed(42)
        wandb.init(project="A2C", entity="rozlvet66")

    def create_model(self) -> nn.Module:
        if isinstance(self.environment.observation_space, gym.spaces.Discrete):
            number_observation_features = self.environment.observation_space.n
        else:
            number_observation_features = self.environment.observation_space.shape[0]

        if isinstance(self.environment.action_space, gym.spaces.Discrete):
            number_actions = self.environment.action_space.n
        else:
            number_actions = self.environment.action_space.shape[0]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNModel(number_observation_features, number_actions, self.linspace).to(device)
        return self.model

    def reinforce(self, env2):
        n_training_episodes = 20
        max_t = 20000
        self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        episodes_rewards = []
        for i_episode in range(1, n_training_episodes + 1):
            saved_log_probs = []
            values = []
            actions = []
            rewards = []
            env2_rewards = []

            obs = self.environment.reset()[0]
            env2.reset()
            for t in range(max_t):
                value, action, log_prob = self.model.act(obs)

                values.append(value)
                actions.append(action)
                saved_log_probs.append(log_prob)

                obs, reward, done, _, _ = self.environment.step(action)
                if t % 100 == 0:
                    print("{action:", action, "reward:", reward, ",observation:", obs,)
                _, rew2, _, _, _ = env2.step(0)
                env2_rewards.append(rew2)
                rewards.append(reward)

                if done:
                    break
            Qval = values[-1].item()
            Qvals = np.zeros_like(rewards)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gamma * Qval
                Qvals[t] = Qval

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            values = torch.FloatTensor(values).to(device)
            Qvals = torch.FloatTensor(Qvals).to(device)
            log_probs = torch.stack(saved_log_probs)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

            episodes_rewards.append(np.mean(rewards))

            wandb.log({"reward": episodes_rewards[-1]})
            wandb.log({"baseline": np.mean(env2_rewards)})
            print('Episode {}\tAverage Reward: {:.2f}'.format(i_episode, episodes_rewards[-1]), "baseline: ", np.mean(env2_rewards))

        return episodes_rewards

    def save(self, path):
        path = path[:-3] + "pth"
        torch.save(self.model.state_dict(), path)


def main():
    environment_config_path = os.path.join("configs", "env.yaml")
    with open(environment_config_path, "r") as f:
        environment_config = yaml.safe_load(f)

    env = make_env('configs/env.yaml', 'train.csv')
    env2 = make_env('configs/env.yaml', 'train.csv')

    algorithm_config_path = os.path.join("configs", "vpg" + ".yaml")
    with open(algorithm_config_path, "r") as f:
        algorithm_config = yaml.safe_load(f)

    vpg = A2C("dummy", env, algorithm_config["model"])
    wandb.login()

    vpg.reinforce(env2)
    model = vpg.model

    logs_path = os.path.join("logs", "vpg", str(int(time.time())))
    os.makedirs(logs_path, exist_ok=True)
