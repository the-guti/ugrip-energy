import time
import os
import argparse
from utils import set_seed, make_env
from algorithms.vpg_discrete_batched import VPG
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import sys
import yaml
#os.chdir('/workspace/UGRIP-ExtraTime/train_scripts')
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("\n\n\n\n=>", os.getcwd())
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env_path", type=str, default="/workspace/UGRIP-ExtraTime/configs/env.yaml")
parser.add_argument("--logs_path", type=str, default="/workspace/UGRIP-ExtraTime/runs/vpg_discrete/")
parser.add_argument("--algorithm", type=str, default="vpg")
parser.add_argument("--total_timesteps", type=int, default=5000000)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--entropy_coef", type=float, default=0.5)
parser.add_argument("--pi_nns", type=int, default=512)
parser.add_argument("--disc_acts", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--n_features", type=float, default=24)
args = parser.parse_args()

# Set seed for reproducibility
set_seed(args.seed)

algorithm_config = {
    "learning_rate": args.learning_rate, 
    "gamma": args.gamma,
    "entropy_coef": args.entropy_coef,
    "batch_size": args.batch_size,
    "clip_range": args.clip_range,
    "disc_acts" : args.disc_acts,
    "pi_nns" : args.pi_nns,
    "n_features" : args.n_features,
    "policy_kwargs": {
        "net_arch": {
            "pi": [args.pi_nns, 2 * args.pi_nns],
        }
    },
    "verbose": 1,
}

lrs = np.linspace(0.000003, 0.03, 10).tolist()

sweep_config = {
    "method": "random", 
    "metric" : {
        "name" : "epoch_average_reward", 
        "goal" : "maximize"
    }, 
    "parameters": {
        "total_timesteps": {"values": [17520*3]},
        "learning_rate": {"values": lrs},
        "gamma" : {"values": [0.99]},
        "batch_size": {"values": [1024]},
        "disc_acts": {"values": [5, 20, 100]},
        "pi_nns": {"values": [1024, 2048, 4096, 8192, 16384]},
        "entropy_coef" : {"values": [0.5]},
        "n_features" :{"values": [1]}, 
        "clip_range" : {"values": [0.7, 0.9, 1]}
    },
}
env, env_config = make_env("/workspace/UGRIP-ExtraTime/configs/env.yaml", "train.csv") 

sweep_id = wandb.sweep(sweep_config, project="ugrip-energy-again")
#sweep_id = 'h6ogjxr8'
def sweep_run():
    # Initialize WandB
    with wandb.init(
        project="ugrip-energy-theafterwork",
        name=args.algorithm + "_" + str(int(time.time())), 
        sync_tensorboard=True,
    ) as run:
        print(run)
        config=wandb.config
        algorithm_config['learning_rate'] = config.learning_rate    
        algorithm_config['gamma'] = config.gamma
        algorithm_config['batch_size'] = config.batch_size
        algorithm_config['entropy_coef'] = config.entropy_coef
        algorithm_config['clip_range'] = config.clip_range
        algorithm_config['disc_acts'] = config.disc_acts
        algorithm_config['pi_nns'] = config.pi_nns
        algorithm_config['n_features'] = config.n_features
        algorithm_config['policy_kwargs'] = None
        algorithm_config['verbose'] = 1

        # Set up logging directory
        logs_path = args.logs_path + run.id + "/"
        os.makedirs(logs_path, exist_ok=True)

        # Initialize the model
        model = VPG("MlpPolicy", env, **algorithm_config, tensorboard_log=logs_path)

        # Train the model
        model.train(
            total_timesteps=config.total_timesteps
        )

        # Save the environment and the model
        model.save(logs_path + "model.pth")
        env.save(logs_path + "env.pkl")

wandb.agent(sweep_id=sweep_id, function=sweep_run, project="ugrip-energy-again")