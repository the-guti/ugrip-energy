import torch
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical
import torch.nn.functional as F
import gymnasium as gym  
import torch.optim as optim
from torch import softmax
from collections import deque
import os 
import yaml
import sys
import time 
# Get the absolute path of the project directory
module_dir = "/MyComputer/MBZUAI Studying material/thursday/ugrip-energy/building_energy_storage_simulation"
# Add the project directory to the Python path
sys.path.insert(0, module_dir)

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
        """
        Args: 
            env (Environment): custom enviroment class, that extends gym.Env
        """
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNModel(number_observation_features, number_actions, self.linspace).to(device)
        #wandb.watch(self.model, log="all", log_freq=10)
        return self.model
    
    def reinforce(self, env2, config=None):
        n_training_episodes=500
        max_t=100000 
        self.create_model()
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
        # Help us to calculate the score during the training
        scores_deque = deque(maxlen=100)
        scores_dequeue_for_wandb = deque(maxlen=100)
        scores = []

        # Line 3 of pseudocode, epochs/ episodes
        for i_episode in range(1, n_training_episodes+1):
            saved_log_probs = []
            actions = []
            rewards = []
            rewards_for_wandb = []
            total_cost = 0
            total_cost2 = 0
            state = self.environment.reset()[0]
            env2.reset()
            #state = state[[0, 3, 18]]
            #state = np.append(state, [0])
            # Line 4 of pseudocode, time steps. 1 loop = 1 ep
            for t in range(max_t):
                action, log_prob = self.model.act(state)
                actions.append(action)
                saved_log_probs.append(log_prob)
                state, reward, done, _ , info= self.environment.step(action)
                _, _, _, _ , info2= env2.step(0)
                #total_cost += info['cost_of_external_generator']
                #total_cost2 += info2['cost_of_external_generator']
                
                #state = state[[0, 3, 18]]
                #state = np.append(state, [t+1])
                #state[0] = state[0] / 20

                advantage = reward
                rewards_for_wandb.append(reward)
                rewards.append(advantage)

                if done:
                    break 

            scores_deque.append(sum(rewards))
            scores_dequeue_for_wandb.append(sum(rewards_for_wandb))
            scores.append(sum(rewards))
            
            # Line 6 of pseudocode: calculate the return
            returns = deque(maxlen=max_t) 
            n_steps = len(rewards) 
            returns = [np.sum(rewards)]* n_steps
            # Compute the discounted returns at each timestep,
            # as 
            #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
            #
            # In O(N) time, where N is the number of time steps
            # (this definition of the discounted return G_t follows the definition of this quantity 
            # shown at page 44 of Sutton&Barto 2017 2nd draft)
            # G_t = r_(t+1) + r_(t+2) + ...
            
            # Given this formulation, the returns at each timestep t can be computed 
            # by re-using the computed future returns G_(t+1) to compute the current return G_t
            # G_t = r_(t+1) + gamma*G_(t+1)
            # G_(t-1) = r_t + gamma* G_t
            # (this follows a dynamic programming approach, with which we memorize solutions in order 
            # to avoid computing them multiple times)
            
            # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
            # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
            
            
            ## Given the above, we calculate the returns at timestep t as: 
            #               gamma[t] * return[t] + reward[t]
            #
            ## We compute this starting from the last timestep to the first, in order
            ## to employ the formula presented above and avoid redundant computations that would be needed 
            ## if we were to do it from first to last.
            
            ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
            ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
            ## a normal python list would instead require O(N) to do this.
            # for t in range(n_steps)[::-1]:
            #     disc_return_t = (returns[0] if len(returns)>0 else 0)
            #     returns.appendleft( gamma*disc_return_t + rewards[t]   )    
                
            ## standardization of the returns is employed to make training more stable
            # eps = np.finfo(np.float32).eps.item()
            ## eps is the smallest representable float, which is 
            # added to the standard deviation of the returns to avoid numerical instabilities        
            returns = torch.tensor(returns)
            # returns = (returns - returns.mean()) / (returns.std() + eps)
            # Line 7:
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(log_prob * disc_return)
            policy_loss = -torch.cat(policy_loss).sum()  
            
            # Line 8: PyTorch prefers gradient descent 
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            average_score =  np.mean(scores_dequeue_for_wandb)
            print('Episode {}\t cost of external generator: {:.2f}'.format(i_episode, average_score))
        
        return scores
    
    
    def save(self, path):
        path = path[:-3] + "pth"
        torch.save(self.model.state_dict(), path)


#main 
#Environment with dataset of 1 day 
environment_config_path = os.path.join("configs", "env.yaml")
with open(environment_config_path, "r") as f:
    environment_config = yaml.safe_load(f)

# Initialize the environment
#env = Environment(dataset="train.csv", **environment_config)
#env2 = Environment(dataset="train.csv", **environment_config)
env = gym.make("CartPole-v1")
env2 = gym.make("CartPole-v1")

#Algorithm config for vpg 
algorithm_config_path = os.path.join("configs", "vpg" + ".yaml")
with open(algorithm_config_path, "r") as f:
    algorithm_config = yaml.safe_load(f)

#create an instance of vpg 
vpg = VPG("dummy", env, algorithm_config["model"])

#do training 
vpg.reinforce(env2)
model = vpg.model

#save model 
logs_path = os.path.join("logs", "vpg", str(int(time.time())))
os.makedirs(logs_path, exist_ok=True)
vpg.save(os.path.join(logs_path, "model.pth"))

    
