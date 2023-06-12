from building_energy_storage_simulation.battery import Battery


class Building:
    """
    Building class.

    Args:
        solar_power_installed (float): The installed peak photovoltaic power in kWp.
        wind_power_installed (float): The installed peak wind power in kW.
        battery_capacity (float): The capacity of the battery in kWh.
        max_battery_charge_per_timestep (float): Maximum amount of energy (kWh) which can be obtained from the battery or which can be used to charge the battery in one time step.
    """

    def __init__(
        self,
        solar_power_installed,
        wind_power_installed,
        battery_capacity,
        initial_state_of_charge,
        max_battery_charge_per_timestep,
    ):
        self.battery = Battery(
            battery_capacity=battery_capacity,
            initial_state_of_charge=initial_state_of_charge,
            max_battery_charge_per_timestep=max_battery_charge_per_timestep,
        )
        self.solar_power_installed = solar_power_installed
        self.wind_power_installed = wind_power_installed
        pass

    def reset(self):
        """
        Resetting the state of the battery by calling `reset()` method from the battery class.
        """
        self.battery.reset()
