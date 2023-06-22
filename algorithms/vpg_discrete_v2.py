import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
module_dir = "/MyComputer/MBZUAI Studying material/thursday/ugrip-energy/building_energy_storage_simulation"
sys.path.insert(0, module_dir)

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
from building_energy_storage_simulation.environment import Environment


class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim, linspace):
        super(NNModel, self).__init__()
        self.linspace = linspace
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return softmax(self.fc4(x), dim=1)

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            action = Categorical(output).sample().item()
            return torch.tensor([[action]])

    def act(self, state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        action_indx = action.item()
        return action_indx, m.log_prob(action)


class VPG:

    def __init__(self, dummy, env, config) -> NNModel:
        self.environment = env
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_prob_actions = []
        self.discounted_rewards = []
        self.losses = []
        self.gamma = 0.95
        self.learning_rate = 0.003
        self.linspace = np.linspace(-1, 1, 20)
        torch.manual_seed(42)

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

    def reinforce(self, env2, config=None):
        n_training_episodes = 500
        max_t = 100000
        self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scores_deque = deque(maxlen=100)
        scores_dequeue_for_wandb = deque(maxlen=100)
        scores = []

        for i_episode in range(1, n_training_episodes + 1):
            saved_log_probs = []
            actions = []
            rewards = []
            rewards_for_wandb = []
            total_cost = 0
            total_cost2 = 0
            state = self.environment.reset()[0]
            env2.reset()

            for t in range(max_t):
                action, log_prob = self.model.act(state)
                actions.append(action)
                saved_log_probs.append(log_prob)
                state, reward, done, _, info = self.environment.step(action)
                _, _, _, _, info2 = env2.step(0)

                advantage = reward
                rewards_for_wandb.append(reward)
                rewards.append(advantage)

                if done:
                    break

            scores_deque.append(sum(rewards))
            scores_dequeue_for_wandb.append(sum(rewards_for_wandb))
            scores.append(sum(rewards))

            returns = deque(maxlen=max_t)
            n_steps = len(rewards)
            returns = [np.sum(rewards)] * n_steps

            returns = torch.tensor(returns)

            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(log_prob * disc_return)
            policy_loss = -torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            average_score = np.mean(scores_dequeue_for_wandb)
            print('Episode {}\t cost of external generator: {:.2f}'.format(i_episode, average_score))

        return scores

    def save(self, path):
        path = path[:-3] + "pth"
        torch.save(self.model.state_dict(), path)


# main
environment_config_path = os.path.join("configs", "env.yaml")
with open(environment_config_path, "r") as f:
    environment_config = yaml.safe_load(f)

env = gym.make("CartPole-v1")
env2 = gym.make("CartPole-v1")

algorithm_config_path = os.path.join("configs", "vpg" + ".yaml")
with open(algorithm_config_path, "r") as f:
    algorithm_config = yaml.safe_load(f)

vpg = VPG("dummy", env, algorithm_config["model"])

vpg.reinforce(env2)
model = vpg.model

logs_path = os.path.join("logs", "vpg", str(int(time.time())))
os.makedirs(logs_path, exist_ok=True)
vpg.save(os.path.join(logs_path, "model.pth"))