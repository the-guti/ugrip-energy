import time
import os
import argparse
from utils import set_seed, make_env
from stable_baselines3 import A2C
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="configs/env.yaml")
    parser.add_argument("--logs_path", type=str, default="runs/a2c/")
    parser.add_argument("--algorithm", type=str, default="a2c")
    parser.add_argument("--total_timesteps", type=int, default=8000000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning_rate", type=float, default=0.0009)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.988)
    parser.add_argument("--vf_coef", type=float, default=0.85)
    parser.add_argument("--qf_nns", type=int, default=256)
    parser.add_argument("--pi_nns", type=int, default=256)
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    algorithm_config = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "gamma": args.gamma,
        "vf_coef": args.vf_coef,
        "policy_kwargs": {
            "net_arch": {
                "pi": [args.pi_nns, 2 * args.pi_nns],
                "qf": [args.qf_nns, 2 * args.qf_nns],
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
        model = A2C("MlpPolicy", env, **algorithm_config, tensorboard_log=logs_path)
        
        # Setup a custom logger
        custom_logger = configure(logs_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(custom_logger)

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