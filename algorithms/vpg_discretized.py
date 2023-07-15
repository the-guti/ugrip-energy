import os
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
import math 
import sys
import wandb 
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from building_energy_storage_simulation.environment import Environment
from train_scripts.utils import make_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class VPG:
    def __init__(self, 
        policy,
        env, 
        learning_rate,
        gamma,
        batch_size,
        entropy_coef,
        clip_range,
        disc_acts,
        pi_nns,
        n_features,
        policy_kwargs,
        verbose, 
        tensorboard_log):

        self.environment = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.clip_range = clip_range
        self.disc_acts = disc_acts
        self.pi_nns = pi_nns
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.n_features = n_features
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_prob_actions = []
        self.discounted_rewards = []
        self.losses = []
        self.linspace = np.linspace(-1, 1, self.disc_acts)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.create_model(self.n_features, self.disc_acts, self.pi_nns)
        torch.manual_seed(42)

    def create_model(self, number_observation_features: int,
        number_actions: int, 
        hidden_layer_features: int) -> nn.Module:
        self.model = nn.Sequential(
            nn.Linear(in_features=number_observation_features,
                    out_features=hidden_layer_features*4),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_features*4,
                    out_features=hidden_layer_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_features,
                    out_features=number_actions),
        ).to(self.device)
        return self.model 
    
    def get_policy(self, observation: np.ndarray) -> Categorical:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        logits = self.model(observation_tensor)

        # Categorical will also normalize the logits for us
        return Categorical(logits=logits), softmax(logits, dim=-1)
        
    def get_action(self, policy: Categorical, probability_distribution):
        action = policy.sample()  # Unit tensor

        # Converts to an int, index of action in linspace 
        action_int = action.item()

        # Get the value of action, in index action_int in discretized action space 
        action_value = self.linspace[action_int]

        # Calculate the log probability of the action, which is required for
        # calculating the loss later
        log_probability_action = policy.log_prob(action)

        return action_value, log_probability_action, probability_distribution[action_int]
    
    def calculate_loss(self, epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> float:
        return -(epoch_log_probability_actions * epoch_action_rewards).mean()
    
    def calc_discount_rewards(self, rewards): 
        discounted_reward = [0] * len(rewards)
        discounted_reward[-1] = rewards[-1]
        for i in reversed(range(len(rewards)-1)):
            discounted_reward[i] = rewards[i] + rewards[i+1] * self.gamma
        return discounted_reward  

    def take_subset_of_obs(self, obs):
        if self.n_features == 1: 
            return obs[[0]]
        elif self.n_features == 3: 
            return obs[[0, 3, 19]]
        elif self.n_features == 7:  
            return obs[[0, 3, 4, 9, 14, 19, 23]]
        else: 
            return obs
    
    def plot(self, loss_over_batches, reward_over_batches, reward_over_batches2):
        fig, axes = plt.subplots(1, 2, figsize=(30, 18))
        axes[0].plot([i.item() for i in loss_over_batches])
        axes[0].set_title('loss')
        axes[1].plot([i for i in reward_over_batches])
        axes[1].plot([i for i in reward_over_batches2], color='red')
        axes[1].set_title('reward')
        fig.suptitle("total_timesteps={},learning_rate={},gamma={},batch_size={},disc_acts={},pi_nns={},entropy_coef={}".format(self.n_steps, 
            self.learning_rate, self.gamma, self.batch_size, self.disc_acts, self.pi_nns, self.entropy_coef), wrap=True)
        plt.savefig('plot_{}.png'.format(wandb.run.name))
        plt.close()
        artifact = wandb.Artifact('plot', type='image')
        artifact.add_file('plot_{}.png'.format(wandb.run.name))
        wandb.run.log_artifact(artifact)
    
    def train(self, total_timesteps): 
        if isinstance(self.environment, VecNormalize): 
            env2, _ = make_env("configs/env.yaml", "train.csv")
        else: 
            with open("configs/env.yaml", "r") as f:
                env_config = yaml.safe_load(f)
            # Initialize the environment
            env2 = Environment(dataset="train.csv", **env_config)

        self.n_steps = total_timesteps
        optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        if isinstance(self.environment, VecNormalize): 
            len_dataset = len(self.environment.venv.envs[0].simulation.electricity_load_profile)
        else: 
            len_dataset = len(self.environment.simulation.electricity_load_profile)
        n_epochs = math.ceil(self.n_steps / len_dataset)
        n_batches = math.floor(len_dataset / self.batch_size)
        remaining = len_dataset % self.batch_size - 5

        reward_over_epochs = []
        reward_over_epochs2 = []
        for epoch in range(n_epochs): 
            loss_over_batches = []
            reward_over_batches = []
            reward_over_batches2 = []
            reward_over_batches_non_norm = []
            reward_over_batches_non_norm2 = []
            observation = self.environment.reset()[0]
            observation = self.take_subset_of_obs(observation)
            env2.reset()
            for batch in range(n_batches + 1):
                reward_batch_non_norm = 0
                reward_batch_non_norm2 = 0 
                batch_rewards = []
                batch_rewards2 = []
                batch_log_prob_actions = []
                batch_prob_actions = []
                len_batch = remaining if batch == n_batches else self.batch_size
                for i in range(len_batch): 
                    policy, probability_distribution = self.get_policy(observation)
                    action, log_probability_action, probability_action = self.get_action(policy, probability_distribution)

                    if isinstance(self.environment, VecNormalize): 
                        action = [action]
                        observation, reward, done, _ = self.environment.step(action) 
                        _, rew2, _, _ = env2.step([0])
                        observation = observation[0]   
                    else: 
                        observation, reward, done, _, _ = self.environment.step(action) 
                        _, rew2, _, _ , _= env2.step(0)               

                    if isinstance(self.environment, VecNormalize): 
                        reward_batch_non_norm += self.environment.get_original_reward()
                        reward_batch_non_norm2 += env2.get_original_reward()   
                    else: 
                        reward_batch_non_norm += reward
                        reward_batch_non_norm2 += rew2          

                    observation = self.take_subset_of_obs(observation)
                    batch_rewards.append(reward)
                    batch_rewards2.append(rew2)
                    batch_log_prob_actions.append(log_probability_action)
                    batch_prob_actions.append(probability_action)

                discounted_rewards = self.calc_discount_rewards(batch_rewards)
                discounted_rewards2 = self.calc_discount_rewards(batch_rewards2)
                batch_log_prob_actions = batch_log_prob_actions[1:]
                discounted_rewards = discounted_rewards[1:]
                batch_prob_actions = batch_prob_actions[1:]

                advantage = [i-j-10000 for i, j in zip(discounted_rewards, discounted_rewards2)]
                #Entropy for regularization -log(p(a|s)) * p(a|s)
                entropy_loss = self.calculate_loss(torch.stack(
                    batch_log_prob_actions),
                    torch.as_tensor(
                    batch_prob_actions, dtype=torch.float32)
                )

                #loss -log(p(a|s)) * R
                loss = self.calculate_loss(torch.stack(
                    batch_log_prob_actions),
                    torch.as_tensor(
                    advantage, dtype=torch.float32)
                )
                loss = loss + self.entropy_coef * entropy_loss * 1000
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)    
                optimizer.step()
                optimizer.zero_grad()

                loss_over_batches.append(loss)
                reward_over_batches.append(discounted_rewards[0])
                reward_over_batches2.append(discounted_rewards2[0])

                reward_over_batches_non_norm.append(reward_batch_non_norm)
                reward_over_batches_non_norm2.append(reward_batch_non_norm2)
            reward_over_epochs.append(np.sum(reward_over_batches_non_norm))
            reward_over_epochs2.append(np.sum(reward_over_batches_non_norm2))
        self.plot(loss_over_batches, reward_over_batches_non_norm, reward_over_batches_non_norm2)
        wandb.log({"epoch_average_reward" : reward_over_epochs[-1]})
        self.save("runs/vpg_discrete/")
        return loss_over_batches, reward_over_epochs, reward_over_epochs2
    
    def save(self, path): 
        logs_path = path + str(int(time.time()))
        os.makedirs(logs_path, exist_ok=True)
        path = logs_path + "/model.pth"
        torch.save(self.model.state_dict(), path)

