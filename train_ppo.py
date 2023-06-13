import time
import yaml
import os
import pandas as pd
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from building_energy_storage_simulation import Environment
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="configs/env.yaml")
    parser.add_argument("--logs_path", type=str, default="runs/ppo/")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--total_timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--use_sde", type=bool, default=False)
    parser.add_argument("--sde_sample_freq", type=int, default=-1)
    parser.add_argument("--vf_nns", type=int, default=32)
    parser.add_argument("--pi_nns", type=int, default=32)
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    algorithm_config = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "clip_range": args.clip_range,
        "use_sde": args.use_sde,
        "sde_sample_freq": args.sde_sample_freq,
        "policy_kwargs": {
            "net_arch": {
                "pi": [args.pi_nns, args.pi_nns],
                "vf": [args.vf_nns, args.vf_nns],
            }
        },
        "verbose": 1,
    }

    # Load environment configuration from file
    with open(args.env_path, "r") as f:
        environment_config = yaml.safe_load(f)

    # Initialize WandB
    run =  wandb.init(
        project="ugrip-energy",
        config={
            "algorithm": args.algorithm,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "algorithm_config": algorithm_config,
            "environment_config": environment_config,
        },
        name="ppo_" + str(int(time.time())), 
        sync_tensorboard=True,
    )

    # Set up logging directory
    logs_path = args.logs_path + run.id + "/"
    os.makedirs(logs_path, exist_ok=True)

    # Initialize the environment
    env = Environment(dataset="train.csv", **environment_config)

    # Wrap the environment with Monitor for logging and DummyVecEnv for normalization
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Initialize the model
    model = PPO("MlpPolicy", env, **algorithm_config, tensorboard_log=logs_path)

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=100,
            model_save_path=logs_path,
            verbose=2,
        )
    )

    # Save the environment
    env.save(logs_path + "env.pkl")

    # Close the environment and the WandB run
    run.finish()

if __name__ == "__main__":
    main()