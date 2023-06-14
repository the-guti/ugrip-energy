import time
import os
import argparse
from utils import set_seed, make_env
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="configs/env.yaml")
    parser.add_argument("--logs_path", type=str, default="runs/ppo/")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--total_timesteps", type=int, default=8000000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--use_sde", type=bool, default=False)
    parser.add_argument("--sde_sample_freq", type=int, default=-1)
    parser.add_argument("--vf_nns", type=int, default=128)
    parser.add_argument("--pi_nns", type=int, default=128)
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

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
                "pi": [args.pi_nns, 2 * args.pi_nns],
                "vf": [args.vf_nns, 2 * args.vf_nns],
            }
        },
        "verbose": 1,
    }

    # Initialize the environment
    env, env_config = make_env(args.env_path, "train.csv")

    # Initialize WandB
    with wandb.init(
        project="ugrip-energy",
        config={
            "algorithm": args.algorithm,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "algorithm_config": algorithm_config,
            "environment_config": env_config,
        },
        name=args.algorithm + "_" + str(int(time.time())), 
        sync_tensorboard=True,
    ) as run:

        # Set up logging directory
        logs_path = args.logs_path + run.id + "/"
        os.makedirs(logs_path, exist_ok=True)

        # Initialize the model
        model = PPO("MlpPolicy", env, **algorithm_config, tensorboard_log=logs_path)

        # Train the model
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=WandbCallback()
        )

        # Save the environment and the model
        model.save(logs_path + "model")
        env.save(logs_path + "env.pkl")

if __name__ == "__main__":
    main()