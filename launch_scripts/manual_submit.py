import datetime
import numpy as np
import time

from simple_slurm import Slurm

# Configure Slurm object

# slurm = Slurm(
#     cpus_per_task=2,
#     mem='40G',
#     qos='cpu-4',
#     partition='cpu',
#     job_name='BCTE',
#     output=f'./logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
#     error=f'./logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.err',
#     time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
# )

slurm = Slurm(
    cpus_per_task=8,
    mem='40G',
    gres='gpu:1',
    qos='gpu-8',
    partition='gpu',
    job_name='BCTE',
    output=f'./slurm_logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    error=f'./slurm_logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.err',
    time=datetime.timedelta(days=0, hours=12, minutes=0, seconds=0),
)


# Run list of experiments

experiments = [
    {
        "algo": "ppo",
        "total_timesteps": 3000000,
        "clip_range":0.1304,
        "gamma":0.915,
        "lr":0.0003854,
        "pi_nns":64,
        "vf_nns":256,
    },
    {
        "algo": "td3",
        "total_timesteps": 3000000,
        "gamma":0.948,
        "lr":0.00278,
        "pi_nns":64,
        "qf_nns":512,
        "tau":0.001479,

    },
    {
        "algo": "ddpg",
        "total_timesteps": 3000000,
        "gamma":0.9821,
        "lr":0.001024,
        "qf_nns":512,
        "pi_nns":256,
        "tau":0.008066,
    },
    {
        "algo": "a2c",
        "total_timesteps": 3000000,
        "gamma":0.9459,
        "lr":0.001893,
        "pi_nns":64,
        "qf_nns":512,
        "vf_coef":0.433,
    },
    {
        "algo": "sac",
    },
    {
        "algo": "vpg",
    }
]

for exp in experiments:

    # exp_name = f"ts_{exp['total_timesteps']}_algo_{exp['algo']}_best"
    
    # print(f"Starting with exp: {exp_name}")

    try:

        if exp['algo'] == "ppo":

            slurm.sbatch(f"python ./train_scripts/ppo.py --total_timesteps {exp['total_timesteps']} --clip_range {exp['clip_range']} --gamma {exp['gamma']} --learning_rate {exp['lr']} --pi_nns {exp['pi_nns']} --vf_nns {exp['vf_nns']}")

        elif exp['algo'] == "td3":

            slurm.sbatch(f"python ./train_scripts/td3.py --gamma {exp['gamma']} --learning_rate {exp['lr']} --pi_nns {exp['pi_nns']} --qf_nns {exp['qf_nns']} --tau {exp['tau']}")

        elif exp['algo'] == "ddpg":

            slurm.sbatch(f"python ./train_scripts/ddpg.py --total_timesteps {exp['total_timesteps']} --gamma {exp['gamma']} --learning_rate {exp['lr']} --pi_nns {exp['pi_nns']} --qf_nns {exp['qf_nns']} --tau {exp['tau']}")

        elif exp['algo'] == "a2c":

            slurm.sbatch(f"python ./train_scripts/a2c.py --total_timesteps {exp['total_timesteps']} --gamma {exp['gamma']} --learning_rate {exp['lr']} --pi_nns {exp['pi_nns']} --qf_nns {exp['qf_nns']} --vf_coef {exp['vf_coef']}")

        elif exp['algo'] == "sac":

            slurm.sbatch(f"python ./train_scripts/sac.py")

        elif exp['algo'] == "vpg":

            slurm.sbatch(f"python ./train_scripts/vpg.py")

    except:

        print("An exception occurred")
                
    time.sleep(np.random.randint(1, 10))

print(f"Finished!")