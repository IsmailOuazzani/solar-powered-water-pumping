import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba

CONSTANT_DEMAND_PATTERN = np.ones(24)
DIURNAL_DEMAND_PATTERN = np.array([
    np.exp(-0.5 * ((i - 6) / 2) ** 2) + np.exp(-0.5 * ((i - 18) / 2) ** 2)
    for i in range(24)
])
CONSTANT_DEMAND_PATTERN_NORMALIZED = CONSTANT_DEMAND_PATTERN / CONSTANT_DEMAND_PATTERN.sum()
DIURNAL_DEMAND_PATTERN_NORMALIZED = DIURNAL_DEMAND_PATTERN / DIURNAL_DEMAND_PATTERN.sum()

def calculate_volume_water_pumped(num_panels, power, time_range, head, pump_eff, inverter_eff, hydraulic_const):
    """
    Calculate the volume of water pumped based on power output.
    """
    return (num_panels * power * time_range * pump_eff * inverter_eff) / (hydraulic_const * head)

def calculate_volume_water_demand(baseline_daily_need: float, time_range_index: pd.DatetimeIndex, pattern: str = "constant") -> pd.Series:
    if pattern == "constant":
        demand_pattern = CONSTANT_DEMAND_PATTERN_NORMALIZED
    elif pattern == "diurnal":
        demand_pattern = DIURNAL_DEMAND_PATTERN_NORMALIZED
    else:
        raise ValueError("Invalid pattern. Use 'constant' or 'diurnal'.")

    # Calculate the demand for each time step
    hourly_demand = baseline_daily_need * demand_pattern
    # Align the demand pattern with the time range index, considering the start time
    start_offset = (time_range_index[0].hour + (time_range_index[0].minute // 60)) % 24
    demand_repeated = np.tile(hourly_demand, (len(time_range_index) + start_offset) // 24 + 1)
    demand_series = pd.Series(demand_repeated[start_offset:start_offset + len(time_range_index)], index=time_range_index)

    return demand_series

def plot_water_simulation(results, time_range, tank_capacity, title, show_percentage=True):
    """
    Plot water simulation for a specified time range with improved clarity for publication.
    
    Parameters:
      - results: DataFrame with a datetime index and columns:
          "water_pumped" (m³ per hour),
          "water_demand" (m³ per hour),
          "water_in_tank" (m³).
      - time_range: A slice or indexer for the time period to plot.
      - tank_capacity: The capacity of the tank (in m³).
      - title: Title of the plot.
      - show_percentage: Boolean flag. If True, display all values as percentages of tank capacity.
    
    In non-percentage mode, the figure is split into two subplots:
      * The top subplot shows the flow data (water pumped and water demand).
      * The bottom subplot shows the volume in tank and the tank capacity.
    """
    
    # Extract data for the specified time range.
    data = results.loc[time_range]
    
    if show_percentage:
        # Convert values to percentages of tank capacity.
        water_pumped_pct = data["water_pumped"] / tank_capacity * 100
        water_demand_pct = data["water_demand"] / tank_capacity * 100
        water_in_tank_pct = data["water_in_tank"] / tank_capacity * 100
        
        # Create a single-axis plot for percentage mode.
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        ax.plot(data.index, water_pumped_pct, label="Water pumped (% of capacity)", linewidth=2, color="blue")
        ax.plot(data.index, water_demand_pct, label="Water demand (% of capacity)", linewidth=2, color="orange")
        ax.plot(data.index, water_in_tank_pct, label="Water in tank (% of capacity)", linewidth=2, color="green")
        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label="Tank capacity (100%)")
        
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.tight_layout()
        plt.savefig(f"outputs/water_in_tank_{time_range}_percentage.png")
        plt.close(fig)
        
    else:
        # Create two subplots: one for flow data and one for tank volume.
        fig, (ax_flow, ax_vol) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
        
        # Top subplot: Flow data (water pumped and water demand).
        ax_flow.plot(data.index, data["water_pumped"], label="Water pumped (m³/h)", linewidth=2, color="blue")
        ax_flow.plot(data.index, data["water_demand"], label="Water demand (m³/h)", linewidth=2, color="orange")
        ax_flow.set_ylabel("Flow (m³/h)", fontsize=12)
        ax_flow.legend(loc="upper left", fontsize=10)
        ax_flow.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Bottom subplot: Tank volume (water in tank) and tank capacity.
        ax_vol.plot(data.index, data["water_in_tank"], label="Water in tank (m³)", linewidth=2, color="green")
        ax_vol.axhline(y=tank_capacity, color='red', linestyle='--', linewidth=2, label="Tank capacity (m³)")
        ax_vol.set_xlabel("Time", fontsize=12)
        ax_vol.set_ylabel("Volume (m³)", fontsize=12)
        ax_vol.legend(loc="upper left", fontsize=10)
        ax_vol.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"outputs/water_in_tank_{time_range}.png")
        plt.close(fig)

@numba.njit
def simulate_tank(u: np.ndarray, tank_capacity: float, initial_tank_level_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized water tank simulation compiled with Numba.
    Computes the water level and water deficit over time.
    """
    n = u.shape[0]
    water_in_tank = np.empty(n + 1)
    water_deficit = np.empty(n + 1)
    water_in_tank[0] = tank_capacity * initial_tank_level_frac
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

def simulate_water_tank(results: pd.DataFrame, tank_capacity: float, initial_tank_level_frac: float ) -> pd.DataFrame:
    """
    Simulate the water tank dynamics using a vectorized approach with Numba.
    Computes the water level and water deficit over time.
    
    Parameters:
        results (pd.DataFrame): DataFrame with columns "water_pumped" and "water_demand".
        tank_capacity (float): Maximum capacity of the water tank (m³).
        initial_tank_level_frac (float): Initial tank level as a fraction of the tank capacity.
    
    Returns:
        pd.DataFrame: Updated DataFrame with "water_in_tank" and "water_deficit" columns.
    """
    # Compute net water input (water pumped minus water demand)
    u = (results["water_pumped"] - results["water_demand"]).values
    water_in_tank, water_deficit = simulate_tank(u, tank_capacity, initial_tank_level_frac)
    # Exclude the initial state to match the DataFrame length
    results["water_in_tank"] = water_in_tank[1:]
    results["water_deficit"] = water_deficit[1:]
    return results
