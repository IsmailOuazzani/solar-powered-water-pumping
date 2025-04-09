"""
For all the variables, we use the following units (unless explicitely stated):
Volume: m³
Energy: Watt
Temperature: Celsius (unfortunate choice due to pvlib)
Cost: USD
"""
from data import import_merra2_dataset, DATASETS
from geography import mask_lon_lat, plot_heatmap, mask_landmass
from pv_system import make_pv_system, appraise_system
from water_system import calculate_volume_water_pumped, calculate_volume_water_demand, simulate_water_tank, plot_water_simulation
import pvlib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss_analysis import lpsp, lpsp_total, clpsp
from time import perf_counter
import seaborn as sns
from pathlib import Path
import xarray as xr
from argparse import ArgumentParser
import calplot
import pickle


import logging
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# TODO: move quite a few of these variables to argparse
COUNTRY="Morocco"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
PATTERN="diurnal"

IPHONE_16_PRO_MAX_CAPACITY = 18 # Wh
SOLAR_PANEL_EFFICIENCY = 0.15 # Lower end, from internet

PUMPING_HEAD = 45  # meters, from Bouzidi paper
PUMP_EFFICIENCY = 0.4 # from Bouzidi paper
INVERTER_EFFICIENCY = 0.95 # from Bouzidi paper
WATER_DEMAND_DAILY = 60 # m³/hour, from Bouzidi paper
WATER_DEMAND_HOURLY = WATER_DEMAND_DAILY/24 
WATER_DEMAND_DAILY_PER_USER = 0.1 # m³/day TODO: justify
NUM_USERS = WATER_DEMAND_DAILY / WATER_DEMAND_DAILY_PER_USER
STORAGE_FACTOR = 0.65  # optimal result in the paper
NUMBER_SOLAR_PANELS = 43  # optimal result in Bouzidi paper
HYDRAULIC_CONSTANT = 2.725

# Optimisation
OPTIM_NUM_PANELS_RANGE = np.linspace(1, 150, 20)
OPTIM_NUM_STORAGE_FACTOR_RANGE = np.linspace(0.01, 3, 20) # TODO: it might become more intuitive to use storage volume instead of storage factor
TARGET_LOSS = 0.0035
SHORTAGE_THRESHOLD = 0.1 # 10% of tank volume


logger.info(f"Config space number of panels: {OPTIM_NUM_PANELS_RANGE}")
logger.info(f"Config space storage factor: {OPTIM_NUM_STORAGE_FACTOR_RANGE}")

logger.info(f"Number users: {NUM_USERS} for a daily demand of {WATER_DEMAND_DAILY} m³/day")

def simulate_at_location( # TODO: return only power series ...
    longitude: float,
    latitude: float,
    solar_radiation_ds: xr.Dataset,
    num_panels: int,
    storage_factor: float,
    initial_tank_level_frac: float,
    pattern: str
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
    results["water_demand"] = calculate_volume_water_demand(
        baseline_daily_need = WATER_DEMAND_DAILY,
        time_range_index=results.index,
        pattern=pattern
    )

    tank_capacity: float = 24 * storage_factor * WATER_DEMAND_HOURLY  # in m³
    results = simulate_water_tank(results, tank_capacity, initial_tank_level_frac)
    return results, tank_capacity


def run_pv_simulation(solar_radiation_ds: xr.Dataset, latitude: float, longitude: float):
    """
    Run the PV simulation for a given geographic coordinate.
    Returns the simulation results DataFrame and the weather DataFrame.
    This function caches the results as a pickle file in the local "./tmp" directory.
    Files are named as "{longitude}_{latitude}.pkl".
    """

    # Set up cache directory
    cache_dir = Path("./tmp")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{longitude}_{latitude}.pkl"

    # If a cached file exists, load and return it.
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            results, weather = pickle.load(f)
        return results, weather

    # Otherwise, run the simulation.
    location_ds = solar_radiation_ds.sel(lat=latitude, lon=longitude, method="nearest")
    pv_system = make_pv_system(latitude=latitude, longitude=longitude)
    weather = location_ds.to_dataframe().rename(columns={"SWGDN": "ghi"})

    # Get solar position and compute dni and dhi.
    solar_position = pvlib.solarposition.get_solarposition(location_ds.time, latitude, longitude)
    weather["dni"] = pvlib.irradiance.disc(
        ghi=weather.ghi,
        solar_zenith=solar_position.zenith,
        datetime_or_doy=weather.index
    )["dni"]
    weather["dhi"] = - np.cos(np.radians(solar_position.zenith)) * weather.dni + weather.ghi

    sim_out = pv_system.run_model(weather).results
    results = pd.DataFrame({"power": sim_out.ac})
    results["power"] = results.power.clip(lower=0).fillna(0)

    # Save the results and weather as a tuple using pickle.
    with open(cache_file, "wb") as f:
        pickle.dump((results, weather), f)
    return results, weather

def evaluate_system(
    results: pd.DataFrame, 
    number_solar_panels: int, 
    storage_factor: float, 
    initial_tank_level_frac: float,
    pattern: str,
) -> float:
    """
    Evaluate system performance (loss function) for given configuration parameters.
    
    This version uses the vectorized simulate_water_tank function.
    """
    # TODO: double check that it is ok to just multiply the power with the number of solar panels
    # might be some important non linearities with the inverter system
    rad: pd.Series = results.power
    time_range: float = 1 # TODO: get rid of this ugly time range logic

    water_pumped: pd.Series = calculate_volume_water_pumped(
        number_solar_panels,
        rad,
        time_range,
        head=PUMPING_HEAD,
        pump_eff=PUMP_EFFICIENCY,
        inverter_eff=INVERTER_EFFICIENCY,
        hydraulic_const=HYDRAULIC_CONSTANT
    )

    tank_capacity = storage_factor * WATER_DEMAND_DAILY

    df = pd.DataFrame({
        "water_pumped": water_pumped,
        "water_demand": calculate_volume_water_demand(
            baseline_daily_need=WATER_DEMAND_DAILY, 
            time_range_index=rad.index,
            pattern=pattern),
    }, index=rad.index)
    df = simulate_water_tank(df, tank_capacity, initial_tank_level_frac) # TODO: this function appears to be called left and right

    return lpsp_total(df["water_deficit"].values, df["water_demand"].values)



def run_optimisation(
    results: pd.DataFrame, 
    panels_range: np.ndarray, 
    storage_range: np.ndarray,
    pattern: str,
    initial_tank_level_frac: float = 1.0, 
    target_loss: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[str, float]]]:
    # TODO: only pass the power column to this to better separate concerns
    """
    Run hyperparameter optimisation for different configurations.
    Returns arrays of losses, costs, Pareto front indices, and the list of hyperparameter configurations.
    """
    hyperparams: list[dict[str, float]] = []
    for num in panels_range:
        for storage in storage_range:
            cost = appraise_system(
                number_solar_panels=int(num),
                tank_capacity=24 * storage * WATER_DEMAND_HOURLY
            ) / NUM_USERS
            hyperparams.append({
                "number_solar_panels": float(num), 
                "storage_factor": float(storage),
                "cost": cost
                })
    hyperparams = sorted(hyperparams, key=lambda x: x["cost"])
    losses: list[float] = [] # TODO: group, with hyperparams, into a Config class
    costs: list[float] = []
    for config in tqdm(hyperparams, desc="Evaluating configurations"):
        loss = evaluate_system(
            results=results, 
            number_solar_panels=int(config["number_solar_panels"]), 
            storage_factor=config["storage_factor"], 
            initial_tank_level_frac=initial_tank_level_frac,
            pattern=pattern)
        logger.debug(f"Config {config} -> loss: {loss}, cost: {cost}")
        losses.append(loss)
        costs.append(config["cost"])
        if target_loss is not None and loss < target_loss:
            break

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
    pareto_indices: list[int],
    hyperparams: list[dict[str, float]]
) -> None:
    """
    Plot contour maps and the Pareto front for the optimisation.
    
    Although the hyperparams (and corresponding losses/costs arrays) are sorted by cost,
    we reconstruct a 2D grid for contour plotting using the original panels and storage ranges.
    """
    grid_losses = np.full((len(panels_range), len(storage_range)), np.nan)
    grid_costs = np.full((len(panels_range), len(storage_range)), np.nan)
    
    # Reconstruct the grid based on the hyperparameters configuration.
    # This is needed as the hyperparams are sorted by cost during the optimisation.
    for k, config in enumerate(hyperparams):
        panel_val = config["number_solar_panels"]
        storage_val = config["storage_factor"]
        # Find indices in the original arrays; use isclose to avoid floating point issues.
        i = int(np.where(np.isclose(panels_range, panel_val))[0][0])
        j = int(np.where(np.isclose(storage_range, storage_val))[0][0])
        grid_losses[i, j] = losses[k]
        grid_costs[i, j] = costs[k]
    
    xs, ys = np.meshgrid(storage_range, panels_range, sparse=False)

    # Plot loss contour using the grid array
    plt.contourf(xs, ys, grid_losses)
    cb = plt.colorbar()
    cb.set_label("Loss (LPSP)")
    plt.xlabel("Storage Factor")
    plt.ylabel("Number of solar panels")
    plt.title("Loss Contour")
    plt.savefig(OUTPUT_DIR / "tradeoff_hourly.png")
    plt.clf()

    # Plot cost contour using the grid array
    plt.contourf(xs, ys, grid_costs)
    cb = plt.colorbar()
    cb.set_label("Cost per community member (USD)")
    plt.xlabel("Storage Factor")
    plt.ylabel("Number of solar panels")
    plt.title("Cost Contour")
    plt.savefig(OUTPUT_DIR / "cost_hourly.png")
    plt.clf()

    # Plot Pareto front in the cost-loss space.
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
    plt.clf()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", 
                        type=str, 
                        default="local", 
                        choices={"local", "global", "global-optim", "background", "demand"},
                        help="Type of simulation to run")
    args = parser.parse_args()

    solar_radiation_ds = import_merra2_dataset(
        DATASETS["M2T1NXRAD_5-2023_only_SWGDN"], 
        variables=["SWGDN"]) # TODO: pass dataset name as argparse argument
    
     # ----------------- Simulation over all points ----------------- #
    if args.type == "global":
        # Mask points outside of selected country
        longitudes = solar_radiation_ds['lon'].values  
        latitudes = solar_radiation_ds['lat'].values 
        start_mask = perf_counter()
        lon_lat_pairs = mask_lon_lat(longitudes,latitudes, 
                                    #  continent="Africa", 
                                    country_name=COUNTRY,
                                     plot=False)
        logging.info(f"Masked {100*(1-len(lon_lat_pairs)/(len(longitudes)*len(latitudes)))}% of data points in {perf_counter()-start_mask}s.")
    
        pv_outputs = []
        pv_outputs_sums = []
        losses_total = []
        shortage_days = []
        final_water_levels_jan1 = [] # TODO: rename
        
        for longitude, latitude in tqdm(lon_lat_pairs):
            logger.debug(f"Simulating system at {longitude}, {latitude}...")
            results, tank_capacity = simulate_at_location(
                solar_radiation_ds=solar_radiation_ds, 
                latitude=latitude, # TODO: change this to pass only location_ds
                longitude=longitude,
                num_panels=NUMBER_SOLAR_PANELS,
                storage_factor=STORAGE_FACTOR,
                initial_tank_level_frac=1.0,
                pattern=PATTERN,
            )

            pv_outputs.append(results)
            pv_outputs_sums.append(results.power.sum())
            final_water_levels_jan1.append(results.loc["2023-12-21 23:30:00", "water_in_tank"])
            logger.debug(f"Evaluating loss function at {longitude}, {latitude}")
            loss = evaluate_system(
                results=results, 
                number_solar_panels=NUMBER_SOLAR_PANELS, 
                storage_factor=STORAGE_FACTOR,
                initial_tank_level_frac=1.0,
                pattern=PATTERN)
            losses_total.append(loss)

            logger.debug(f"Storing results for {longitude}, {latitude}")
            volume_at_end_of_day = results[results.index.strftime('%H:%M') == "23:30"].water_in_tank
            volume_at_end_of_day = volume_at_end_of_day[:-1] # TODO: fix datasets to not have this
            num_shortage_days = (volume_at_end_of_day < SHORTAGE_THRESHOLD * tank_capacity).sum()
            shortage_days.append(num_shortage_days)

        first_data_point_pv = pv_outputs[0]

        # Print the water demand from the first 24 hours of the first data point
        first_24_hours_water_demand = first_data_point_pv["water_demand"].iloc[:24]

        plot_water_simulation(first_data_point_pv, "2023-12-21", tank_capacity, "") # TODO:move this to local section
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

        start_simul = perf_counter()
        results, weather = run_pv_simulation(solar_radiation_ds, target_latitude, target_longitude)
        logger.info(f"Simulated PV system at {target_longitude}, {target_latitude} in {perf_counter()-start_simul:.2f} seconds.")
        
        losses, costs, pareto_indices, hyperparams = run_optimisation(
            results, 
            OPTIM_NUM_PANELS_RANGE, 
            OPTIM_NUM_STORAGE_FACTOR_RANGE, 
            pattern=PATTERN,
            initial_tank_level_frac=1.0, 
            target_loss=None
        )
        plot_optimisation_results(
            panels_range=OPTIM_NUM_PANELS_RANGE, 
            storage_range=OPTIM_NUM_STORAGE_FACTOR_RANGE, 
            losses=losses, 
            costs=costs, 
            pareto_indices=pareto_indices,
            hyperparams=hyperparams,
            )
        
        # ----------------- Analysis of best configuration solution at single location ----------------- #
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
            storage_factor=best_config["storage_factor"],
            initial_tank_level_frac=1.0,
            pattern=PATTERN
        )
        end_of_day_tank_capacity = results[results.index.strftime('%H:%M') == "23:30"].water_in_tank
        end_of_day_tank_capacity = end_of_day_tank_capacity[:-1] # TODO: fix datasets to not have this
        end_of_day_tank_capacity_percentage = 100 * end_of_day_tank_capacity / tank_capacity
        calplot.calplot(end_of_day_tank_capacity_percentage,
                        suptitle="Percentage of tank capacity at the end of the day",
                        cmap="RdYlBu",
                        colorbar=True)
        plt.savefig(OUTPUT_DIR / "water_in_tank_end_of_day.png")
        plt.clf()

        num_shortage_days = (end_of_day_tank_capacity < SHORTAGE_THRESHOLD * tank_capacity).sum()
        logger.info(f"Number of days with water shortage: {num_shortage_days}")

        # Analyse sensibility to initial condition in tank fullness level
        initial_conditions = np.linspace(0.0, 1.0, 10)
        shortage_days = []
        shortage_hours = []
        losses = []
        for initial_condition in initial_conditions:
            results, tank_capacity = simulate_at_location(
                solar_radiation_ds=solar_radiation_ds, 
                latitude=target_latitude, 
                longitude=target_longitude,
                num_panels=best_config["number_solar_panels"],
                storage_factor=best_config["storage_factor"],
                initial_tank_level_frac=initial_condition,
                pattern=PATTERN
            )
            volume_at_end_of_day = results[results.index.strftime('%H:%M') == "23:30"].water_in_tank
            volume_at_end_of_day = volume_at_end_of_day[:-1]
            num_shortage_days = (volume_at_end_of_day < SHORTAGE_THRESHOLD * tank_capacity).sum()
            loss = evaluate_system(
                results=results, 
                number_solar_panels=best_config["number_solar_panels"], 
                storage_factor=best_config["storage_factor"],
                initial_tank_level_frac=initial_condition,
                pattern=PATTERN)
            shortage_days.append(num_shortage_days)
            shortage_hours.append((results["water_in_tank"] < SHORTAGE_THRESHOLD * tank_capacity).sum())
            losses.append(loss)


        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.plot(initial_conditions, shortage_days, color='tab:blue', label="Shortage Days")
        ax1.set_xlabel("Initial tank level fraction")
        ax1.set_ylabel("Number of days with water shortage")
        ax1.set_title("Shortage Days vs Initial Tank Level")
        ax1.grid()
        ax2.plot(initial_conditions, shortage_hours, color='tab:green', label="Shortage Hours")
        ax2.set_xlabel("Initial tank level fraction")
        ax2.set_ylabel("Number of hours with water shortage")
        ax2.set_title("Shortage Hours vs Initial Tank Level")
        ax2.grid()
        ax3.plot(initial_conditions, losses, color='tab:red', label="Loss Function")
        ax3.set_xlabel("Initial tank level fraction")
        ax3.set_ylabel("Loss function (LPSP)")
        ax3.set_title("Loss Function vs Initial Tank Level")
        ax3.grid()

        fig.tight_layout()
        plt.savefig(OUTPUT_DIR / "sensitivity_initial_tank_level_combined.png")
        plt.clf()


    # ----------------- Optimisation for all the points  ----------------- #
    if args.type == "global-optim":
        start_mask = perf_counter()
        longitudes = solar_radiation_ds['lon'].values  
        latitudes = solar_radiation_ds['lat'].values 
        lon_lat_pairs = mask_lon_lat(
            longitudes,
            latitudes, 
            # country_name=COUNTRY, 
            continent="Africa",
            plot=False)
        logging.info(f"Masked {100*(1-len(lon_lat_pairs)/(len(longitudes)*len(latitudes)))}% of data points in {perf_counter()-start_mask}s.")

        best_configs = []

        logging.info("Starting global optimization")
        for longitude, latitude in tqdm(lon_lat_pairs):
            results, tank_capacity = simulate_at_location(
                solar_radiation_ds=solar_radiation_ds, 
                latitude=latitude, 
                longitude=longitude,
                num_panels=NUMBER_SOLAR_PANELS,
                storage_factor=STORAGE_FACTOR,
                initial_tank_level_frac=1.0,
                pattern=PATTERN
            )
            losses_arr, costs_arr, pareto_indices, hyperparams = run_optimisation(
                results=results, 
                panels_range=OPTIM_NUM_PANELS_RANGE, 
                storage_range=OPTIM_NUM_STORAGE_FACTOR_RANGE,
                target_loss=TARGET_LOSS,
                pattern=PATTERN
            )
            valid_pareto_indices = [i for i in pareto_indices if losses_arr[i] < TARGET_LOSS]
            best_config_index = np.argmin(costs_arr[valid_pareto_indices])
            best_index = valid_pareto_indices[best_config_index]
            best_config = hyperparams[best_index]

            best_config["longitude"] = longitude
            best_config["latitude"] = latitude
            best_config["loss"] = losses_arr[best_index]
            best_config["cost"] = costs_arr[best_index]
            best_config["pattern"] = PATTERN

            # TODO: need to cache the following if we want to use it
            # location_ds = solar_radiation_ds.sel(lat=latitude, lon=longitude, method="nearest")
            # avg_solar_rad = float(location_ds.SWGDN.mean().values)
            # var_solar_rad = float(location_ds.SWGDN.var().values)
            # best_config["avg_solar_rad"] = avg_solar_rad
            # best_config["var_solar_rad"] = var_solar_rad
            
            results_best, tank_capacity = simulate_at_location(
                solar_radiation_ds=solar_radiation_ds, 
                latitude=latitude, 
                longitude=longitude,
                num_panels=best_config["number_solar_panels"],
                storage_factor=best_config["storage_factor"],
                initial_tank_level_frac=1.0,
                pattern=PATTERN
            )
            volume_at_end_of_day = results_best[results_best.index.strftime('%H:%M') == "23:30"].water_in_tank
            volume_at_end_of_day = volume_at_end_of_day[:-1]
            num_shortage_days = (volume_at_end_of_day < SHORTAGE_THRESHOLD * tank_capacity).sum()
            best_config["shortage_days"] = num_shortage_days

            shortage_hours = (results_best["water_in_tank"] < SHORTAGE_THRESHOLD * tank_capacity).sum()
            best_config["shortage_hours"] = shortage_hours

            best_config["avg_tank_volume"] = results_best["water_in_tank"].mean()
            best_config["var_tank_volume"] = results_best["water_in_tank"].var()

            best_configs.append(best_config)

        # Save results
        best_configs_df = pd.DataFrame(best_configs)    
        best_configs_df.to_csv(OUTPUT_DIR / f"best_configs_{PATTERN}_.csv")

        plot_heatmap(
            lon_lat_pairs=lon_lat_pairs,
            values=np.array([config["cost"] for config in best_configs]),
            output_file=OUTPUT_DIR / "best_config_costs.png",
            legend="Best Config Cost per community member (USD)",
            hue_style="green"
        ) # TODO: the lack of diversity of best config suggests that 
        # either the loss function is not sensitive enough or the range of hyperparameters too wide, not
        # focused on the region of interest.
        plot_heatmap(
            lon_lat_pairs=lon_lat_pairs,
            values=np.array([config["shortage_days"] for config in best_configs]),
            output_file=OUTPUT_DIR / "best_config_shortage_days.png",
            legend="Best Config Number of days with water shortage",
            hue_style="red"
        )

        # Analyse correlations 
        corr = best_configs_df.corr()
        plt.figure(figsize=(20,20), dpi=300) 
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(OUTPUT_DIR / "global_correlation_matrix_best_config.png")

        # Plot distribution of best configs (solar panels, storage factor)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        sns.histplot(best_configs_df["number_solar_panels"], ax=axes[0])
        sns.histplot(best_configs_df["storage_factor"], ax=axes[1])
        axes[0].set_title("Distribution of best number of solar panels")
        axes[1].set_title("Distribution of best storage factor")
        plt.savefig(OUTPUT_DIR / "global_best_configs_distribution.png")

        # TODO: analyse this for all configs, not just the best ones    

    if args.type == "background":
        # Plot the total SWGDN, cumulative summed over the year, for the entire world.
        # Convert Wh/m² (assuming hourly values) to the number of iPhone 16 Pro Max charges per m².
        total_swgdn = solar_radiation_ds["SWGDN"].sum(dim="time")
        total_swgdn_iphone_charges = total_swgdn / IPHONE_16_PRO_MAX_CAPACITY
        total_swgdn_iphone_charges = total_swgdn_iphone_charges * SOLAR_PANEL_EFFICIENCY

        lon_lat_pairs, land_mask = mask_landmass(
            lon=solar_radiation_ds.lon.values,
            lat=solar_radiation_ds.lat.values
        )
        land_values = total_swgdn_iphone_charges.values[land_mask]  
        plot_heatmap(
            lon_lat_pairs=lon_lat_pairs,
            values=land_values,
            output_file=OUTPUT_DIR / "total_swgdn_iphone_charges.png",
            legend="Yearly iPhone 16 Pro Max Charges per m²",
            hue_style="orange"
        )

        plt.figure(figsize=(10, 6))
        plt.hist(total_swgdn.values.flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title("Histogram of Total SWGDN (Yearly Cumulative)")
        plt.xlabel("Total SWGDN (Wh/m²)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(OUTPUT_DIR / "total_swgdn_histogram.png")
        plt.clf()

    if args.type == "demand":
        # For a single location, compare the diurnal and constant demand patterns.
        latitude = 27.8667
        longitude = -0.2833

        results, tank_capacity = simulate_at_location(
            solar_radiation_ds=solar_radiation_ds, 
            latitude=latitude, 
            longitude=longitude,
            num_panels=NUMBER_SOLAR_PANELS,
            storage_factor=STORAGE_FACTOR,
            initial_tank_level_frac=1.0,
            pattern="constant"
        )
        losses_cte, costs_cte, pareto_indices_cte, hyperparams_cte = run_optimisation(
            results=results, 
            panels_range=OPTIM_NUM_PANELS_RANGE, 
            storage_range=OPTIM_NUM_STORAGE_FACTOR_RANGE,
            target_loss=TARGET_LOSS,
            pattern="constant"
        )
        losses_diurnal, costs_diurnal, pareto_indices_diurnal, hyperparams_diurnal = run_optimisation(
            results=results, 
            panels_range=OPTIM_NUM_PANELS_RANGE, 
            storage_range=OPTIM_NUM_STORAGE_FACTOR_RANGE,
            target_loss=TARGET_LOSS,
            pattern="diurnal"
        )
        # Compare the Pareto fronts
        plt.scatter(costs_cte, losses_cte, label="Constant Demand", marker="o")
        plt.scatter(costs_diurnal, losses_diurnal, label="Diurnal Demand", marker="x")
        plt.xlabel("Cost per community member (USD)")
        plt.ylabel("Loss of Power Supply Probability (LPSP)")
        plt.title("Pareto Front Comparison")
        plt.legend()
        plt.grid()
        plt.savefig(OUTPUT_DIR / "pareto_front_comparison.png")        
        
        

        

        
