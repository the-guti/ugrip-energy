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
# import wandb
from utils import *
# Get the absolute path of the project directory
module_dir = "/MyComputer/MBZUAI Studying material/thursday/ugrip-energy/building_energy_storage_simulation"
# Add the project directory to the Python path
sys.path.insert(0, module_dir)

from building_energy_storage_simulation.environment import Environment


class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim, linspace):
        super(NNModel, self).__init__()
        self.linspace = linspace
        self.fc1_critic = nn.Linear(24, 512)
        self.fc2_critic = nn.Linear(512, 1024)
        self.fc3_critic = nn.Linear(1024, 512)
        self.fc4_critic = nn.Linear(512, 1)

        self.fc1_actor = nn.Linear(24, 512)
        self.fc2_actor = nn.Linear(512, 1024)
        self.fc3_actor = nn.Linear(1024, 512)
        self.fc4_actor = nn.Linear(512, 15)

    def forward_critic(self, x):
        x = torch.relu(self.fc1_critic(x))
        x = torch.relu(self.fc2_critic(x))
        x = torch.relu(self.fc3_critic(x))
        return self.fc4_critic(x)

    def forward_actor(self, x):
        x = torch.relu(self.fc1_actor(x))
        x = torch.relu(self.fc2_actor(x))
        x = torch.relu(self.fc3_actor(x))
        return self.fc4_actor(x)

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
        m = Categorical(logits=probs)
        categorical_sampling = m.sample()
        action_indx = categorical_sampling.item()
        action = self.linspace[action_indx]
        log_probs = m.log_prob(categorical_sampling)
        return value, action, log_probs


class A2C:
    def __init__(self, dummy, env, config):
        self.environment = env
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_prob_actions = []
        self.discounted_rewards = []
        self.losses = []
        self.gamma = 0.95
        self.learning_rate = 0.0007
        self.linspace = np.linspace(-0.99, 0.99, 15)
        torch.manual_seed(42)

    def create_model(self) -> nn.Module:
        if isinstance(self.environment[0].observation_space, gym.spaces.Discrete):
            number_observation_features = self.environment[0].observation_space.n
        else:
            number_observation_features = self.environment[0].observation_space.shape[0]

        if isinstance(self.environment[0].action_space, gym.spaces.Discrete):
            number_actions = self.environment[0].action_space.n
        else:
            number_actions = self.environment[0].action_space.shape[0]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNModel(number_observation_features, number_actions, self.linspace).to(device)
        return self.model

    def reinforce(self, env2):
        n_training_episodes = 20
        self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        episodes_rewards = []
        for i_episode in range(1, n_training_episodes + 1):
            obs = self.environment[0].reset()[0]
            env2[0].reset()

            for _ in range(5):
                saved_log_probs = []
                values = []
                actions = []
                rewards = []
                env2_rewards = []
                for t in range(3500):
                    value, action, log_prob = self.model.act(obs)

                    values.append(value)
                    actions.append(action)
                    saved_log_probs.append(log_prob)

                    obs, reward, done, _ = self.environment[0].step([action])
                    _, rew2, _, _ = env2[0].step([0])
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
                values.requires_grad = True

                Qvals = torch.FloatTensor(Qvals).to(device)
                Qvals.requires_grad = True
                Qvals = Qvals.reshape((1, -1))[0]

                saved_log_probs = saved_log_probs[1:]
                Qvals = Qvals[1:]
                values = values[1:]

                log_probs = [i.item() for i in saved_log_probs]
                log_probs = torch.FloatTensor(log_probs).to(device)
                log_probs.requires_grad = True

                advantage = Qvals - values
                actor_loss = (-log_probs * advantage).mean()
                critic_loss = 0.5 * advantage.pow(2).mean()
                ac_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                ac_loss.backward()
                self.optimizer.step()

                episodes_rewards.append(np.mean(rewards))

            print('\n\nEpisode{}\tAverage Reward: {:.9f}'.format(i_episode, episodes_rewards[-1]), "baseline: ", np.mean(env2_rewards), "\n\n")

        return episodes_rewards

    def save(self, path):
        path = path[:-3] + "pth"
        torch.save(self.model.state_dict(), path)


# main
# Environment with dataset of 1 day
environment_config_path = os.path.join("configs", "env.yaml")
with open(environment_config_path, "r") as f:
    environment_config = yaml.safe_load(f)

env = make_env('configs/env.yaml', 'train.csv')
env2 = make_env('configs/env.yaml', 'train.csv')

algorithm_config_path = os.path.join("configs", "vpg" + ".yaml")
with open(algorithm_config_path, "r") as f:
    algorithm_config = yaml.safe_load(f)

vpg = A2C("dummy", env, algorithm_config["model"])

vpg.reinforce(env2)
model = vpg.model

logs_path = os.path.join("logs", "vpg", str(int(time.time())))
os.makedirs(logs_path, exist_ok=True)
vpg.save(os.path.join(logs_path, "model.pth"))