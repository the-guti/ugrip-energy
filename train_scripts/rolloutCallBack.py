from stable_baselines3.common.callbacks import BaseCallback
import random 
class RollOutCallBack(BaseCallback): 
    def __init__(self, verbose=0): 
        super(RollOutCallBack, self).__init__(verbose)

    def _on_rollout_start(self) -> None:
        print("\n\n_on_rollout_start")
        envs = self.training_env.envs
        for env in envs: 
            env.env.simulation.start_index = random.randint(0, 17520 - 130)
            env.env.simulation.initial_state_of_charge = random.uniform(0, 1)
        print("Env specs: ", f"start_index:{envs[0].env.simulation.start_index}", "\n\n")
        print("Env specs: ", f"initial_state_of_charge:{envs[0].env.simulation.initial_state_of_charge}", "\n\n")

    
    def _on_step(self) -> bool:

        print(self.num_timesteps)

        return True 