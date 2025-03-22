"""
For all the variables, we use the following units (unless explicitely stated):
Volume: m³
Energy: Watt
Temperature: Celsius (unfortunate choice due to pvlib)
Cost: USD
"""
from data import import_merra2_dataset, DATASETS
from geography import mask_lon_lat, plot_heatmap
from pv_system import make_pv_system, appraise_system
from water_system import calculate_volume_water_pumped, calculate_volume_water_demand, simulate_water_tank, plot_water_simulation
import pvlib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss_analysis import lpsp, lpsp_total, clpsp
from time import perf_counter
from pathlib import Path
import xarray as xr
from argparse import ArgumentParser
import calplot

import logging
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# TODO: it might become more intuitive to use storage volume instead of storage factor


COUNTRY="Burundi"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PUMPING_HEAD = 45  # meters, from Bouzidi paper
PUMP_EFFICIENCY = 0.4 # from Bouzidi paper
INVERTER_EFFICIENCY = 0.95 # from Bouzidi paper
WATER_DEMAND_HOURLY = 60/24  # m³/hour, from Bouzidi paper
STORAGE_FACTOR = 0.65  # optimal result in the paper
NUMBER_SOLAR_PANELS = 43  # optimal result in Bouzidi paper
HYDRAULIC_CONSTANT = 2.725

# Optimisation
OPTIM_NUM_PANELS_RANGE = np.linspace(1, 100, 10)
OPTIM_NUM_STORAGE_FACTOR_RANGE = np.linspace(0.01, 2, 10)
TARGET_LOSS = 0.05

def simulate_at_location(
    longitude: float,
    latitude: float,
    solar_radiation_ds: xr.Dataset,
    num_panels: int,
    storage_factor: float,
) -> tuple[pd.DataFrame, float]:
    """
    Simulate the PV and water system for a single location.
    Returns the results DataFrame with additional water simulation columns and the tank capacity.
    """
    results, _ = run_pv_simulation(solar_radiation_ds, latitude, longitude)
    results["water_pumped"] = calculate_volume_water_pumped(
        num_panels,
        results.power,
        time_range=1,
        head=PUMPING_HEAD,
        pump_eff=PUMP_EFFICIENCY,
        inverter_eff=INVERTER_EFFICIENCY,
        hydraulic_const=HYDRAULIC_CONSTANT
    )
    results["water_demand"] = calculate_volume_water_demand(WATER_DEMAND_HOURLY, time_range=1)

    tank_capacity: float = 24 * storage_factor * WATER_DEMAND_HOURLY  # in m³
    results = simulate_water_tank(results, tank_capacity)
    return results, tank_capacity

def run_pv_simulation(solar_radiation_ds: xr.Dataset, latitude: float, longitude: float):
    """
    Run the PV simulation for a given geographic coordinate.
    Returns the simulation results DataFrame.
    """
    #TODO: can this be cached?

    # Select the data at the location
    location_ds = solar_radiation_ds.sel(lat=latitude, lon=longitude, method="nearest")
    pv_system = make_pv_system(latitude=latitude, longitude=longitude)
    weather = location_ds.to_dataframe().rename(columns={"SWGDN": "ghi"})

    # Get solar position and compute dni and dhi
    solar_position = pvlib.solarposition.get_solarposition(location_ds.time, latitude, longitude)
    weather["dni"] = pvlib.irradiance.disc(ghi=weather.ghi,
                                            solar_zenith=solar_position.zenith,
                                            datetime_or_doy=weather.index)["dni"]
    weather["dhi"] = - np.cos(np.radians(solar_position.zenith)) * weather.dni + weather.ghi

    sim_out = pv_system.run_model(weather).results
    results = pd.DataFrame({"power": sim_out.ac})
    # clip the power to zero, as negative power does not make sense?
    results["power"] = results.power.clip(lower=0).fillna(0)
    return results, weather


def evaluate_system(
    results: pd.DataFrame, number_solar_panels: int, storage_factor: float
) -> float:
    """
    Evaluate system performance (loss function) for given configuration parameters.
    """
    # TODO: double check that it is ok to just multiply the power with the number of solar panels
    rad: pd.Series = results.power  # hourly power values
    time_range: float = 1
    water_pumped: pd.Series = calculate_volume_water_pumped(
        number_solar_panels,
        rad,
        time_range,
        head=PUMPING_HEAD,
        pump_eff=PUMP_EFFICIENCY,
        inverter_eff=INVERTER_EFFICIENCY,
        hydraulic_const=HYDRAULIC_CONSTANT
    )
    water_demand: list[float] = [calculate_volume_water_demand(WATER_DEMAND_HOURLY, time_range)] * len(rad)
    tank_capacity: float = 24 * storage_factor * WATER_DEMAND_HOURLY

    # TODO: replace this with simulate_water_tank function?
    water_in_tank: list[float] = [tank_capacity]
    water_deficit: list[float] = [0]
    for i in range(1, len(rad)):
        new_level: float = water_in_tank[-1] + water_pumped.iloc[i] - water_demand[i]
        if new_level < 0:
            water_deficit.append(abs(new_level))
            constrained_level: float = 0
        else:
            constrained_level = min(new_level, tank_capacity)
            water_deficit.append(0)
        water_in_tank.append(constrained_level)
    return lpsp_total(water_deficit, water_demand)

def run_optimisation(
    results: pd.DataFrame, panels_range: np.ndarray, storage_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[str, float]]]:
    """
    Run hyperparameter optimisation for different configurations.
    Returns arrays of losses, costs, Pareto front indices, and the list of hyperparameter configurations.
    """
    hyperparams: list[dict[str, float]] = []
    for num in panels_range:
        for storage in storage_range:
            hyperparams.append({"number_solar_panels": float(num), "storage_factor": float(storage)})

    losses: list[float] = []
    costs: list[float] = []
    for config in tqdm(hyperparams, desc="Evaluating configurations"):
        loss = evaluate_system(results, int(config["number_solar_panels"]), config["storage_factor"])
        cost = appraise_system(
            number_solar_panels=int(config["number_solar_panels"]),
            tank_capacity=24 * config["storage_factor"] * WATER_DEMAND_HOURLY)
        logger.info(f"Config {config} -> loss: {loss}, cost: {cost}")
        losses.append(loss)
        costs.append(cost)

    losses_arr: np.ndarray = np.array(losses)
    costs_arr: np.ndarray = np.array(costs)
    
    # Identify Pareto front
    pareto_indices: list[int] = []
    for i in range(len(losses_arr)):
        dominated: bool = any(
            (losses_arr[j] <= losses_arr[i] and costs_arr[j] < costs_arr[i]) or 
            (losses_arr[j] < losses_arr[i] and costs_arr[j] <= costs_arr[i])
            for j in range(len(losses_arr)) if j != i
        )
        if not dominated:
            pareto_indices.append(i)
    return losses_arr, costs_arr, pareto_indices, hyperparams

def plot_optimisation_results(
    panels_range: np.ndarray,
    storage_range: np.ndarray,
    losses: np.ndarray,
    costs: np.ndarray,
    pareto_indices: list[int]
) -> None:
    """
    Plot contour maps and the Pareto front for the optimisation.
    """
    xs, ys = np.meshgrid(storage_range, panels_range, sparse=False)

    # Plot loss contour
    plt.contourf(xs, ys, losses.reshape(xs.shape))
    cb = plt.colorbar()
    cb.set_label("Loss (LPSP)")
    plt.xlabel("Storage Factor")
    plt.ylabel("Number of solar panels")
    plt.title("Loss Contour")
    plt.savefig(OUTPUT_DIR / "tradeoff_hourly.png")
    plt.clf()

    # Plot cost contour
    plt.contourf(xs, ys, costs.reshape(xs.shape))
    cb = plt.colorbar()
    cb.set_label("Cost per community member (USD)")
    plt.xlabel("Storage Factor")
    plt.ylabel("Number of solar panels")
    plt.title("Cost Contour")
    plt.savefig(OUTPUT_DIR / "cost_hourly.png")
    plt.clf()

    # Plot Pareto front
    plt.scatter(costs, losses, alpha=0.5, label="All configurations")
    pareto_costs: np.ndarray = costs[pareto_indices]
    pareto_losses: np.ndarray = losses[pareto_indices]
    sorted_indices: np.ndarray = np.argsort(pareto_costs)
    plt.plot(pareto_costs[sorted_indices], pareto_losses[sorted_indices],
             color="red", marker="o", label="Pareto front")
    plt.xlabel("Cost per community member (USD)")
    plt.ylabel("Loss of Power Supply Probability (LPSP)")
    plt.title("Pareto Front")
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_DIR / "pareto_front_cost_vs_performance.png")
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", type=str, default="global", help="Type of simulation to run: global or local")
    args = parser.parse_args()

    solar_radiation_ds = import_merra2_dataset(
        DATASETS["M2T1NXRAD_5-2023_only_SWGDN"], 
        variables=["SWGDN"]) # TODO: pass dataset name as argparse argument
    
    if args.type == "global":
        # Mask points outside of selected country
        longitudes = solar_radiation_ds['lon'].values  
        latitudes = solar_radiation_ds['lat'].values 
        start_mask = perf_counter()
        lon_lat_pairs = mask_lon_lat(longitudes,latitudes, country_name=COUNTRY, plot=False)
        logging.info(f"Masked {100*(1-len(lon_lat_pairs)/(len(longitudes)*len(latitudes)))}% of data points in {perf_counter()-start_mask}s.")
    

        # ----------------- Simulation over all points ----------------- #
        pv_outputs = []
        pv_outputs_sums = []
        losses_total = []
        final_water_levels_jan1 = [] # TODO: rename

        
        for longitude, latitude in tqdm(lon_lat_pairs):
            logging.info(f"Simulating system at {longitude}, {latitude}...")
            results, tank_capacity = simulate_at_location(
                solar_radiation_ds=solar_radiation_ds, 
                latitude=latitude, 
                longitude=longitude,
                num_panels=NUMBER_SOLAR_PANELS,
                storage_factor=STORAGE_FACTOR
            )

            pv_outputs.append(results)
            pv_outputs_sums.append(results.power.sum())
            final_water_levels_jan1.append(results.loc["2023-12-21 23:30:00", "water_in_tank"])

            loss = evaluate_system(
                results=results, 
                number_solar_panels=NUMBER_SOLAR_PANELS, 
                storage_factor=STORAGE_FACTOR)
            losses_total.append(loss)

        first_data_point_pv = pv_outputs[0]
        plot_water_simulation(first_data_point_pv, "2023-12-21", tank_capacity, "")
        plot_heatmap(lon_lat_pairs=lon_lat_pairs, 
                    values=np.array(pv_outputs_sums), 
                    output_file="outputs/total_power.png", 
                    legend="Total power generated by the pv system [Wh]",
                    hue_style="orange")    
        plot_heatmap(
            lon_lat_pairs=lon_lat_pairs,
            values=100*np.array(final_water_levels_jan1)/tank_capacity,
            output_file="outputs/water_in_tank_end_january_1st.png",
            legend="Fill Level Percentage"
        )
        plot_heatmap(
            lon_lat_pairs=lon_lat_pairs,
            values=np.array(losses_total),
            output_file="outputs/lossfct.png",
            legend="Loss function (Loss of Power Supply Probability)",
            hue_style="red"
        )


    # ----------------- Optimisation at single point  ----------------- #
    if args.type == "local":
        logging.info("Starting local optimization")
        
        # Target location
        target_latitude = 27.8667
        target_longitude = -0.2833 
        location_radiation_ds = solar_radiation_ds.sel(lat=target_latitude, lon=target_longitude, method='nearest')
        pv_system = make_pv_system(latitude=target_latitude, longitude=target_longitude)
        weather = location_radiation_ds.to_dataframe().rename(columns={"SWGDN": "ghi"}) # for compatibility with pvlib
        solar_position = pvlib.solarposition.get_solarposition(location_radiation_ds.time, target_latitude, target_longitude)
        weather["dni"] = pvlib.irradiance.disc(
            ghi=weather.ghi, 
            solar_zenith=solar_position.zenith, 
            datetime_or_doy=weather.index)["dni"] #TODO: try other models for dni
        weather["dhi"] =  - np.cos(np.radians(solar_position.zenith)) * weather.dni + weather.ghi # GHI = DHI + DNI * cos(zenith) https://www.researchgate.net/figure/Equation-of-calculating-GHI-using-DNI-and-DHI_fig1_362326479#:~:text=The%20quantity%20of%20solar%20radiation,)%20%2BDHI%20%5B12%5D%20.
        logging.info(f"Simulating system at {target_longitude}, {target_latitude}...")

        start_simul = perf_counter()
        sim_out = pv_system.run_model(weather).results
        results = pd.DataFrame({"power":sim_out.ac})
        results.power = results.power.clip(lower=0).fillna(0) 
        logger.info(f"simulated pv system in {perf_counter()-start_simul}")
        
        losses, costs, pareto_indices, hyperparams = run_optimisation(results, OPTIM_NUM_PANELS_RANGE, OPTIM_NUM_STORAGE_FACTOR_RANGE)
        plot_optimisation_results(
            panels_range=OPTIM_NUM_PANELS_RANGE, 
            storage_range=OPTIM_NUM_STORAGE_FACTOR_RANGE, 
            losses=losses, 
            costs=costs, 
            pareto_indices=pareto_indices)
        
        # ----------------- Analysis of best configuration at single location ----------------- #
        # Best configuration is the cheapest one with a loss below the target
        # Filter the Pareto indices to those that meet the loss criteria
        valid_pareto_indices = [i for i in pareto_indices if losses[i] < TARGET_LOSS]
        
        best_config_index = np.argmin(costs[valid_pareto_indices])
        best_index = valid_pareto_indices[best_config_index]
        best_config = hyperparams[best_index]
        logger.info(f"Best configuration: {best_config}, loss: {losses[best_index]}, cost: {costs[best_index]}")
        results, tank_capacity = simulate_at_location(
            solar_radiation_ds=solar_radiation_ds, 
            latitude=target_latitude, 
            longitude=target_longitude,
            num_panels=best_config["number_solar_panels"],
            storage_factor=best_config["storage_factor"]
        )
        end_of_day_tank_capacity = results[results.index.strftime('%H:%M') == "23:30"].water_in_tank
        end_of_day_tank_capacity = end_of_day_tank_capacity[:-1] # TODO: fix datasets to not have this
        end_of_day_tank_capacity_percentage = 100 * end_of_day_tank_capacity / tank_capacity
        calplot.calplot(end_of_day_tank_capacity_percentage,
                        suptitle="Percentage of tank capacity at the end of the day",
                        cmap="RdYlBu",
                        colorbar=True)
        plt.savefig(OUTPUT_DIR / "water_in_tank_end_of_day.png")


        
