
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Policy_Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy_Network, self).__init__()
        self.input = nn.Linear(state_size, 16)
        self.fc = nn.Linear(16, 16)
        self.output = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        x = self.output(x)
        x = torch.sigmoid(x)
        #x= torch.tanh(x)
        return x

class VPG:
    def __init__(self, env, gamma=0.95, learning_rate=0.0001, steps_per_epoch=100, epochs=10): #learning rate 0.001
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.policy_net = Policy_Network(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.states = []
        self.actions = []
        self.rewards = []
        
    def get_action(self, state):
        #normalized_state = torch.FloatTensor(state) / torch.max(torch.FloatTensor(state))
        #normalized_state = normalized_state.unsqueeze(0)  # Add batch dimension
        n_state = torch.FloatTensor(state).unsqueeze(0)
        policy_output = self.policy_net(n_state)
        return policy_output.item()

    def store_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        states = torch.FloatTensor(self.states)
        actions = torch.FloatTensor(self.actions)
        discounted_rewards = torch.tensor(self.discount_rewards(self.rewards), dtype=torch.float32)

        policy_output = self.policy_net(states)

        log_probs = -actions * torch.log(policy_output)
        policy_loss = torch.sum(log_probs * discounted_rewards)

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
        n_episode = 0
        max_num_episodes = 600
        averageRewardsPerEpisode = []
        rewards = []
        while n_episode < max_num_episodes:
            state = env.reset()[0]
            episode_rewards = []
            while True:
                action = self.get_action(state)
                new_state, reward, done, _ , info = env.step(action)
             
                self.store_experience(state, action, reward)
                episode_rewards.append(reward)
                state = new_state
                if done:
                    break
            rewards.append(episode_rewards)
            self.learn()
            print(f"Episode {n_episode}: Average reward = {np.mean(rewards[n_episode])}")

            averageRewardsPerEpisode.append([n_episode, np.mean(rewards[n_episode])])
            n_episode += 1

        env.close()
        return rewards, averageRewardsPerEpisode
