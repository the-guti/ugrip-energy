from building_energy_storage_simulation.building import Building
from building_energy_storage_simulation.utils import load_profile


class Simulation:
    """
    Simulation that wires the building, electricity load, and solar generation profile together.

    :param electricity_load_profile_file_name: Path to csv file containing electric load profile.
    :type electricity_load_profile_file_name: str
    :param solar_generation_profile_file_name: Path to csv file containing solar energy generation profile. Note that
        the profile is in W per kWp of solar power installed. The actual solar generation is determined by
        multiplication with the `solar_power_installed`.
    :type solar_generation_profile_file_name: str
    :param external_generation_profile_file_name: Path to csv file containing external energy generation profile and cost.
    :type external_generation_profile_file_name: str
    :param solar_power_installed: The installed peak photovoltaic power in kWp.
    :type solar_power_installed: float
    :param battery_capacity: The capacity of the battery in kWh.
    :type battery_capacity: float
    :param max_battery_charge_per_timestep: Maximum amount of energy (kWh) which can be obtained from the battery or
        which can be used to charge the battery in one time step.
    :type max_battery_charge_per_timestep: float
    :param sell_back_price_rate: Price rate at which excess energy is sold back. In $/kWh.
    :type sell_back_price_rate: float
    """

    def __init__(self,
                 dataset,
                 battery_capacity,
                 initial_state_of_charge,
                 solar_power_installed,
                 max_battery_charge_per_timestep,
                 sell_back_price_rate):
        self.building = Building(solar_power_installed=solar_power_installed,
                                 battery_capacity=battery_capacity,
                                 initial_state_of_charge=initial_state_of_charge,
                                 max_battery_charge_per_timestep=max_battery_charge_per_timestep)

        self.electricity_load_profile = load_profile(dataset, 'load')
        self.solar_generation_profile = load_profile(dataset, 'solar')
        self.external_generation_profile = load_profile(dataset, 'price')
        self.sell_back_price_rate = sell_back_price_rate
        assert len(self.solar_generation_profile) == len(self.electricity_load_profile) == len(self.external_generation_profile), \
            "Solar generation profile, electricity load profile, and external generation profile must be of the same length."
        # Solar Generation Profile is in W per 1KW of Solar power installed
        self.solar_generation_profile = self.solar_generation_profile * self.building.solar_power_installed / 1000
        self.step_count = 0
        self.start_index = 0
        pass

    def reset(self):
        """
        1. Resets the state of the building by calling the `reset()` method from the building class.
        2. Resets the `step_count` to 0. The `step_count` is used for temporal orientation in the electricity
           load and solar generation profile.
        """

        self.building.reset()
        self.step_count = 0
        pass

    def simulate_one_step(self, amount: float) -> float:
        """
        Performs one simulation step by:
            1. Charging or discharging the battery depending on the amount.
            2. Calculating the amount of energy consumed by the building in this time step.
            3. Trimming the amount of energy to 0, in case it is negative.
            4. Calculating the amount of excess energy which is considered lost.
            5. Increasing the step counter.

        :param amount: Amount of energy to be stored or retrieved from the battery. In kWh.
        :type amount: float
        :returns:
            Tuple of:
                1. Amount of energy consumed in this time step. This is calculated by: `battery_energy`
                   + `electricity_load` - `solar_generation`. The excess energy can be sold with the `sell_back_price_rate`.
                2. Amount of excess energy.
                3. Cost of using external generator.
                4. Revenue from selling excess energy.

        :rtype: (float, float, float, float)
        """
        electricity_load_of_this_timestep = self.electricity_load_profile[self.start_index + self.step_count]
        solar_generation_of_this_timestep = self.solar_generation_profile[self.start_index + self.step_count]
        external_generation_cost_of_this_timestep = self.external_generation_profile[self.start_index + self.step_count]

        electricity_consumed_for_battery = self.building.battery.use(amount)
        electricity_consumption = electricity_consumed_for_battery + electricity_load_of_this_timestep - \
                                  solar_generation_of_this_timestep
        excess_energy = 0
        external_generator_energy = 0
        cost_of_external_generator = 0
        revenue_from_excess_energy = 0
        if electricity_consumption < 0:
            excess_energy = -1 * electricity_consumption
            electricity_consumption = 0
            revenue_from_excess_energy = excess_energy * external_generation_cost_of_this_timestep * self.sell_back_price_rate
        else:
            external_generator_energy = electricity_consumption
            cost_of_external_generator = external_generation_cost_of_this_timestep * external_generator_energy
        self.step_count += 1
        return electricity_consumption, excess_energy, cost_of_external_generator, revenue_from_excess_energy