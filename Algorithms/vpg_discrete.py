import torch
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical
import gymnasium as gym  
import torch.optim as optim
from torch import softmax
import os
import zipfile

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim): 
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 20)
    
    def forward(self, x): 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return softmax(self.fc3(x), dim=1)
    
    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            action = Categorical(output).sample().item()
            if(action <= 10): 
                action *= -1
            else: 
                action -= 10
            action *= 0.1
            return torch.tensor([[action]])

    
class VPG: 

    def __init__(self, dummy, env, config) -> NNModel:
        """
        Args: 
            env (Environment): custom enviroment class, that extends gym.Env
        """
        self.environment = env
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_prob_actions = []
        self.normalized_rewards = []
        self.discounted_rewards = []
        self.discount_rewards_normalized = []
        self.losses = []    
        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']


    def create_model(self) -> nn.Module:
        """Create the MLP model

        Args:
            number_observation_features (int): Number of features in the (flat)
            observation tensor
            number_actions (int): Number of actions

        Returns:
            nn.Module: Simple MLP model
        """
        
        #Adapting with CardPole environment
        if isinstance(self.environment.observation_space, gym.spaces.Discrete): 
            number_observation_features = self.environment.observation_space.n
        else: 
            number_observation_features = self.environment.observation_space.shape[0]

        if isinstance(self.environment.action_space, gym.spaces.Discrete): 
            number_actions = self.environment.action_space.n
        else: 
            number_actions = self.environment.action_space.shape[0]

        #NN
        self.model = NNModel(number_observation_features, number_actions)
        return self.model
    
    def get_policy(self, observation: np.ndarray):
        """Get the policy from the model, for a specific observationlearn more about deep reinforcement learning
            observation (np.ndarray): Environment observation

        Returns:
            Categorical: Multinomial distribution parameterized by model logits
        """ 
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        action_distribution = self.model(observation_tensor)
        action_distribution = Categorical(action_distribution)

        action = action_distribution.sample()        
        # Converts to an int, as this is what Gym environments require
        action_int = action.item() 

        log_probability_action = action_distribution.log_prob(action)
        return action_int, log_probability_action
    
    def learn_step(self): 
        discounted_reward_normalized = self.discount_rewards(self.normalized_rewards)
        discounted_reward = self.discount_rewards(self.rewards)
        epoch_loss = self.calculate_loss(torch.stack(
            self.log_prob_actions),
            torch.as_tensor(
            discounted_reward_normalized, dtype=torch.float32)
        )
        mean = np.mean(np.asarray(discounted_reward))
        mean_normalized = np.mean(np.asarray(discounted_reward_normalized))
        self.discounted_rewards.append(mean)
        self.discount_rewards_normalized.append(mean_normalized)
        self.losses.append(epoch_loss)

        self.optimizer.zero_grad()
        epoch_loss.backward()
        self.optimizer.step()

        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.normalized_rewards = []

    def discount_rewards(self, rewards):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        discounted_rewards = torch.zeros_like(rewards_tensor)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = running_sum * 0.95 + rewards_tensor[t]
            discounted_rewards[t] = running_sum
        return discounted_rewards.detach().numpy().tolist()

    
    def calculate_loss(self, epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> float:
        """Calculate the 'loss' required to get the policy gradient

        Formula for gradient at
        https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient

        Note that this isn't really loss - it's just the sum of the log probability
        of each action times the episode return. We calculate this so we can
        back-propagate to get the policy gradient.

        Args:
            epoch_log_probability_actions (torch.Tensor): Log probabilities of the
                actions taken
            epoch_action_rewards (torch.Tensor): Rewards for each of these actions

        Returns:
            float: Pseudo-loss
        """
        return -(epoch_log_probability_actions * epoch_action_rewards).mean()

    def store_experience(self, observation, action, log_prob_action, normalized_reward, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.log_prob_actions.append(log_prob_action)
        self.normalized_rewards.append(normalized_reward)    
        self.rewards.append(reward)    

    def learn(self, config) -> None:
        """Train a Vanilla Policy Gradient model"""
        self.total_timesteps = config['total_timesteps']
        self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        steps = 0
        while steps < self.total_timesteps: 
                observation = self.environment.reset()
                if steps > len(self.environment.envs[0].simulation.solar_generation_profile) - 200: 
                    self.environment.envs[0].simulation.start_index = 0 
                done = False
                for i in range(self.batch_size):
                    action, log_prob_action = self.get_policy(observation)
                    if(action <= 10): 
                        action *= -1
                    else: 
                        action -= 10
                    action *= 0.1
                    action = [action]
                    observation, reward, done, info = self.environment.step(action)
                    self.store_experience(observation, action, log_prob_action, reward, 0)
                self.learn_step()
                steps += self.batch_size

    
    def save(self, path):
        path = path[:-3] + "pth"
        torch.save(self.model.state_dict(), path)
    

        