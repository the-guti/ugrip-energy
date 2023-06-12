

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import Categorical 
import random
import gym
from collections import deque

class Policy_Network(nn.Module): #Policy network 
    def __init__(self, state_size, action_size):
        super(Policy_Network, self).__init__()
        self.input = nn.Linear(state_size, 16)
        self.fc = nn.Linear(16, 16)
        self.output = nn.Linear(16, action_size)

    def forward(self, x): #Returns {action_size} probability values in a tensor
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        x = self.output(x)
        x = F.softmax(x)
        return x
class VPG:
    def __init__(self, env, gamma=0.95, learning_rate=0.001, steps_per_epoch=100, epochs=10 ):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.states = []
        self.actions = []
        self.rewards = []
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs     
        self.policy_net = Policy_Network(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def get_action(self, state): 
        state = torch.FloatTensor(state)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action

    def store_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        discounted_rewards = torch.tensor(self.discount_rewards(self.rewards), dtype=torch.float32)
    
        #Calculating gradient 
        probs = self.policy_net(states)
        sampler = Categorical(probs)
        log_probs = -sampler.log_prob(actions)   
        policy_loss = torch.sum(log_probs * discounted_rewards) 
        #Optimizing the network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


        self.states = []
        self.actions = []
        self.rewards = []
    def discount_rewards(self, rewards):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        discounted_rewards = torch.zeros_like(rewards_tensor)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = running_sum * self.gamma + rewards_tensor[t]
            discounted_rewards[t] = running_sum
        return discounted_rewards.detach().numpy().tolist()
    def train(self):
        env = self.env
        render_rate = 100
        n_episode = 1
        max_num_episodes = 1000
        returns = deque(maxlen=1000)
        data = []
        while n_episode < max_num_episodes:
            state = env.reset()

            while True:

                action = self.get_action(state)
                new_state, reward, done, info = env.step(action.item())
                self.store_experience(state, action, reward)
                state = new_state
                if done:
                    break
            returns.append(np.sum(self.rewards))

            self.learn()
            
            data.append([n_episode, np.mean(returns)])
            n_episode += 1
        env.close()
        return returns, data
def trainAndPlot(env, agent):
  #env = gym.make("CartPole-v1")
  #agent = VPG(env)

  returns, data = agent.train()

  episodes = [row[0] for row in data]
  returns = [row[1] for row in data]

  plt.plot(episodes, returns)
  plt.xlabel('Episode')
  plt.ylabel('Average Return')
  plt.title(f'Episode vs Average Return using our VPG agent in {env.spec.id} env')
  plt.show()
