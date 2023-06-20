import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    """Actor network for the Actor-Critic agent."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        """Forward pass of the actor network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            mu (torch.Tensor): Mean of the action distribution.
            log_std (torch.Tensor): Log standard deviation of the action distribution.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        return mu, log_std


class Critic(nn.Module):
    """Critic network for the Actor-Critic agent."""

    def __init__(self, state_dim, hidden_size=265):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Forward pass of the critic network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            value (torch.Tensor): Estimated state value.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value


class ACAgent:
    """Actor-Critic agent using the Advantage Actor-Critic algorithm."""

    def __init__(self, state_dim, action_dim, lr_actor=1e-9, lr_critic=0.0001, gamma=0.95):
        """
        Initialize the AC agent.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            lr_actor (float): Learning rate for the actor network.
            lr_critic (float): Learning rate for the critic network.
            gamma (float): Discount factor for future rewards.
        """
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        """Select an action using the actor network.

        Args:
            state (np.ndarray): Current state.

        Returns:
            action (np.ndarray): Selected action.
        """
        state = torch.FloatTensor(state)
        mu, log_std = self.actor(state)
        std = torch.exp(log_std) + 1e-5
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.tanh(action)
        return action.detach().numpy()

    def update(self, state, action, next_state, reward, done):
        """Update the actor and critic networks.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Selected action.
            next_state (np.ndarray): Next state.
            reward (float): Reward received from the environment.
            done (bool): Whether the episode is finished.
        """
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        mask = torch.FloatTensor([1 - done])

        value = self.critic(state)
        next_value = self.critic(next_state)

        target = reward + self.gamma * next_value * mask
        advantage = target - value

        # Critic loss
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        mu, log_std = self.actor(state)
        std = torch.exp(log_std) + 1e-5
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action)
        actor_loss = -(log_prob * advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

