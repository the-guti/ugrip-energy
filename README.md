# Building Energy Storage Simulation

This repository focuses on applying Reinforcement Learning (RL) to simulate and control a building's energy storage system. The goal is to achieve optimal control over energy storage (in the form of a battery) and solar/wind energy systems, minimizing the costs associated with buying electricity from the grid and reducing CO<sub>2</sub> emissions.

We have utilized the code from [this repository](https://github.com/tobirohrer/building-energy-storage-simulation). We would like to express our gratitude to the author for their valuable work.

## Repository Organization

The repository is structured into the following directories:

- `algorithms`: Contains the implementations of various RL algorithms.
- `building_energy_storage`: Contains the environment code.
  - `data`: Stores all time series data used.
- `configs`: Includes YAML files for environment configurations and WandB sweeps.
- `notebooks`: Contains various notebooks.
- `train_scripts`: Contains scripts for training different algorithms.
- `test.ipynb`: Notebook for retrieving test set scores.

## Installation

```
pip install -r requirements.txt
```

## Usage

1. Sign in to your WandB account.
```
wandb login
```
2. Set up a sweep and launch the agent for a specific `algorithm`.
```
wandb sweep configs/algorithm.yaml
wandb agent username/ugrip-energy/sweep_id
```
3. Use the provided training scripts in the `train_scripts` folder to train an RL algorithm.
```
python train_scripts/algorithm.py
```
4. Use the `test.ipynb` notebook, input the relative path to the model to evaluate the algorithm on the test set and compute the percentage of money saved.

## Contributing

We warmly welcome contributions to this project. If you have any ideas, suggestions, or improvements, please feel free to submit a pull request or open an issue.