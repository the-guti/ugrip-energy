import random
import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from building_energy_storage_simulation import Environment


def set_seed(seed):
    """
    Set the seed for all random number generators.

    Args:
        seed (int): The seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    return None


def make_env(env_path, dataset):
    """
    Initialize the environment.

    Args:
        env_path (str): Path to the environment configuration file.
        dataset (str): Path to the dataset file.

    Returns:
        env (gym.Env): The environment.
    """
    # Load environment configuration from file
    with open(env_path, "r") as f:
        env_config = yaml.safe_load(f)

    # Initialize the environment
    env = Environment(dataset=dataset, **env_config)

    # Wrap the environment with Monitor for logging and DummyVecEnv for normalization
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env, env_config