import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba

def calculate_volume_water_pumped(num_panels, power, time_range, head, pump_eff, inverter_eff, hydraulic_const):
    """
    Calculate the volume of water pumped based on power output.
    """
    return (num_panels * power * time_range * pump_eff * inverter_eff) / (hydraulic_const * head)

def calculate_volume_water_demand(hourly_need, time_range):
    """
    Calculate the water demand over a given time range.
    """
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
    plt.ylabel("Water (m³)")
    plt.savefig("outputs/water_in_tank.png")
    plt.close()

@numba.njit
def simulate_tank(u: np.ndarray, tank_capacity: float):
    """
    Vectorized water tank simulation compiled with Numba.
    Computes the water level and water deficit over time.
    """
    n = u.shape[0]
    water_in_tank = np.empty(n + 1)
    water_deficit = np.empty(n + 1)
    water_in_tank[0] = tank_capacity
    water_deficit[0] = 0.0
    for i in range(n):
        new_level = water_in_tank[i] + u[i]
        if new_level < 0:
            water_deficit[i + 1] = -new_level  # record deficit as positive value
            water_in_tank[i + 1] = 0.0
        else:
            water_deficit[i + 1] = 0.0
            water_in_tank[i + 1] = new_level if new_level < tank_capacity else tank_capacity
    return water_in_tank, water_deficit

def simulate_water_tank(results: pd.DataFrame, tank_capacity: float) -> pd.DataFrame:
    """
    Simulate the water tank dynamics using a vectorized approach with Numba.
    Computes the water level and water deficit over time.
    
    Parameters:
        results (pd.DataFrame): DataFrame with columns "water_pumped" and "water_demand".
        tank_capacity (float): Maximum capacity of the water tank (m³).
    
    Returns:
        pd.DataFrame: Updated DataFrame with "water_in_tank" and "water_deficit" columns.
    """
    # Compute net water input (water pumped minus water demand)
    u = (results["water_pumped"] - results["water_demand"]).values
    water_in_tank, water_deficit = simulate_tank(u, tank_capacity)
    # Exclude the initial state to match the DataFrame length
    results["water_in_tank"] = water_in_tank[1:]
    results["water_deficit"] = water_deficit[1:]
    return results
