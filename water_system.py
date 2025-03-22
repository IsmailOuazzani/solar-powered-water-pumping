import matplotlib.pyplot as plt
import pandas as pd


def calculate_volume_water_pumped(num_panels, power, time_range, head, pump_eff, inverter_eff, hydraulic_const):
    # TODO: how many solar panels can I fit per inverter?
    return (num_panels * power * time_range * pump_eff * inverter_eff) / (hydraulic_const * head) # TODO: am I allowed to just multiply with num solar panels?


def calculate_volume_water_demand(hourly_need, time_range):
    return hourly_need * time_range

def plot_water_simulation(results, time_range, tank_capacity, title):
    """
    Plot water simulation for a specified time range.
    """
    results.loc[time_range, "water_pumped"].plot()
    results.loc[time_range, "water_demand"].plot()
    results.loc[time_range, "water_in_tank"].plot()

    plt.axhline(y=tank_capacity, color='r', linestyle='--', label="Tank capacity")
    plt.legend(["Water pumped", "Water Demand", "Water in tank", "Tank capacity"])
    plt.title(title)
    plt.ylabel("Water (m^3)")
    plt.savefig("outputs/water_in_tank.png")

def simulate_water_tank(results: pd.DataFrame, tank_capacity: float) -> pd.DataFrame:
    """
    Given a DataFrame 'results' containing water pumped and water demand columns,
    compute the water level in the tank and the water deficit over time.
    """
    water_in_tank: list[float] = [tank_capacity]  # initial condition: full tank
    water_deficit: list[float] = [0]

    for i in range(1, len(results)):
        new_level: float = water_in_tank[-1] + results["water_pumped"].iloc[i] - results["water_demand"].iloc[i]
        if new_level < 0:
            water_deficit.append(abs(new_level))
            constrained_level: float = 0
        else:
            constrained_level = min(new_level, tank_capacity)
            water_deficit.append(0)
        water_in_tank.append(constrained_level)

    results["water_in_tank"] = water_in_tank
    results["water_deficit"] = water_deficit
    return results