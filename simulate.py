"""
For all the variables, we use the following units (unless explicitely stated):
Volume: m³
Energy: Watt
Temperature: Celsius (unfortunate choice due to pvlib)
Cost: USD
"""
from data import import_merra2_dataset, DATASETS
from geography import mask_lon_lat, plot_heatmap
from pvsystem import make_pv_system
import pvlib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss_analysis import lpsp, lpsp_total, clpsp
from time import perf_counter
import seaborn as sns
from pathlib import Path

import logging
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)




pumping_head = 45  # meters, from Bouzidi paper
pump_efficiency = 0.4 # from Bouzidi paper
inverter_efficiency = 0.95 # from Bouzidi paper
WATER_DEMAND_HOURLY = 60/24  # m³/hour, from Bouzidi paper
STORAGE_FACTOR = 0.65  # optimal result in the paper
COST_PER_M3_TANK = 285 # From Ahmed and Demirci
NUMBER_SOLAR_PANELS = 43  # optimal result in Bouzidi paper
COST_PER_PANEL = 330 # From Bouzidi paper  
COST_POWER_INVERTER = 411
COST_PUMP = 750
HYDRAULIC_CONSTANT = 2.725

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


if __name__ == "__main__":
    solar_radiation_ds = import_merra2_dataset(
        DATASETS["M2T1NXRAD_5-2023_only_SWGDN"], variables=["SWGDN"]) # TODO: pass dataset name as argparse argument
    
    lon = solar_radiation_ds['lon'].values  # Replace 'lon' with your longitude coordinate name
    lat = solar_radiation_ds['lat'].values  # Replace 'lat' with your latitude coordinate name


    start_mask = perf_counter()
    lon_lat_pairs = mask_lon_lat(lon,lat, country_name="Algeria", plot=False)
    logging.info(f"Masked {100*(1-len(lon_lat_pairs)/(len(lon)*len(lat)))}% of data points in {perf_counter()-start_mask}s.")
 
    pv_outputs = []
    pv_outputs_sums = []
    losses_total = []

    Path("outputs").mkdir(exist_ok=True)


    final_water_levels_jan1 = []
    for longitude, latitude in tqdm(lon_lat_pairs):
        logging.info(f"Simulating system at {longitude}, {latitude}...")
        location_radiation_ds = solar_radiation_ds.sel(lat=latitude, lon=longitude, method=None)

        pv_system = make_pv_system(latitude=latitude, longitude=longitude)
        weather = location_radiation_ds.to_dataframe()
        weather = weather.rename(columns={"SWGDN": "ghi"}) # for compatibility with pvlib
        solar_position = pvlib.solarposition.get_solarposition(location_radiation_ds.time, latitude, longitude)
        weather["dni"] = pvlib.irradiance.disc(ghi=weather.ghi, solar_zenith=solar_position.zenith, datetime_or_doy=weather.index)["dni"] #TODO: try other models for dni
        weather["dhi"] =  - np.cos(np.radians(solar_position.zenith)) * weather.dni + weather.ghi # GHI = DHI + DNI * cos(zenith) https://www.researchgate.net/figure/Equation-of-calculating-GHI-using-DNI-and-DHI_fig1_362326479#:~:text=The%20quantity%20of%20solar%20radiation,)%20%2BDHI%20%5B12%5D%20.

        sim_out = pv_system.run_model(weather).results
        results = pd.DataFrame({"power":sim_out.ac})
        # clip the power to zero, as negative power does not make sense?
        results.power = results.power.clip(lower=0)
        results.power = results.power.fillna(0) # Don't need to do this when using other solar panels ...


        results["water_pumped"] = calculate_volume_water_pumped(
            NUMBER_SOLAR_PANELS,
            results.power,
            time_range=1,
            head=pumping_head,
            pump_eff=pump_efficiency,
            inverter_eff=inverter_efficiency,
            hydraulic_const=HYDRAULIC_CONSTANT
        )
        results["water_demand"] = calculate_volume_water_demand(WATER_DEMAND_HOURLY, time_range=1)

        tank_capacity = 24 * STORAGE_FACTOR * WATER_DEMAND_HOURLY  # in m^3
        water_in_tank = [tank_capacity]  # Start with a full tank
        water_deficit = [0]

        for i in range(1, len(results)):
            new_water_level = water_in_tank[-1] + results["water_pumped"].iloc[i] - results["water_demand"].iloc[i]
            if new_water_level < 0:
                water_deficit.append(abs(new_water_level))
                constrained_water_level = 0
            else:
                constrained_water_level = min(new_water_level, tank_capacity)
                water_deficit.append(0)
            water_in_tank.append(constrained_water_level)

        results["water_in_tank"] = water_in_tank
        results["water_deficit"] = water_deficit

        # loss = lpsp(results["water_deficit"], results["water_demand"])
        # losses.append(loss)
        losses_total.append(lpsp_total(results["water_deficit"], results["water_demand"]))

        pv_outputs.append(results)
        pv_outputs_sums.append(results.power.sum())
        final_water_levels_jan1.append(results.loc["2023-12-21 23:30:00", "water_in_tank"])


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


    # Optimise system behavior
    start_ds = perf_counter()
    longitude, latitude = lon_lat_pairs[0]
    

    # Target location
    latitude = 27.8667
    longitude = -0.2833 
    location_radiation_ds = solar_radiation_ds.sel(lat=latitude, lon=longitude, method='nearest')
    pv_system = make_pv_system(latitude=latitude, longitude=longitude)

    pv_system = make_pv_system(latitude=latitude, longitude=longitude)
    weather = location_radiation_ds.to_dataframe()
    weather = weather.rename(columns={"SWGDN": "ghi"}) # for compatibility with pvlib
    solar_position = pvlib.solarposition.get_solarposition(location_radiation_ds.time, latitude, longitude)
    weather["dni"] = pvlib.irradiance.disc(ghi=weather.ghi, solar_zenith=solar_position.zenith, datetime_or_doy=weather.index)["dni"] #TODO: try other models for dni
    weather["dhi"] =  - np.cos(np.radians(solar_position.zenith)) * weather.dni + weather.ghi # GHI = DHI + DNI * cos(zenith) https://www.researchgate.net/figure/Equation-of-calculating-GHI-using-DNI-and-DHI_fig1_362326479#:~:text=The%20quantity%20of%20solar%20radiation,)%20%2BDHI%20%5B12%5D%20.
    logger.info(f"Prepared radiation dataset in {perf_counter() - start_ds}")

    start_simul = perf_counter()
    sim_out = pv_system.run_model(weather).results
    results = pd.DataFrame({"power":sim_out.ac})
    # clip the power to zero, as negative power does not make sense?
    results.power = results.power.clip(lower=0)
    results.power = results.power.fillna(0) # Don't need to do this when using other solar panels ...

    logger.info(f"simulated pv system in {perf_counter()-start_simul}")


    hyperparams: list[dict] = []
    number_solar_panels_test = np.linspace(1,100,40)
    storage_factor_test = np.linspace(0.01,2,40) # TODO: might be some problem in how this is indexed...
    # TODO: include cost

    for i in range(len(number_solar_panels_test)):
        for j in range(len(storage_factor_test)):
            hyperparams.append(
                {
                    "results": results, #TODO: see if I can pass this later instead of here
                    "number_solar_panels": number_solar_panels_test[i],
                    "storage_factor": storage_factor_test[j]
                }
            )

    logger.info(f"Testing {len(hyperparams)} configurations.")

    def evaluate_sys(
            results: pd.DataFrame,
            number_solar_panels: int,
            storage_factor: int) -> float:
        
        # rad = results.resample("1D").sum().power # weird reslts when agregating over a day ...
        rad = results.power
        time_range = 1
        
        water_pumped = calculate_volume_water_pumped(
        number_solar_panels,
        rad,
        time_range=time_range,
        head=pumping_head,
        pump_eff=pump_efficiency,
        inverter_eff=inverter_efficiency,
        hydraulic_const=HYDRAULIC_CONSTANT
        )      

        water_demand = [calculate_volume_water_demand(WATER_DEMAND_HOURLY, time_range=time_range)]*len(rad)


        tank_capacity = 24 * storage_factor * WATER_DEMAND_HOURLY 
        water_in_tank = [tank_capacity]  # Start with a full tank
        water_deficit = [0]

        for i in range(1, len(rad)):
            new_water_level = water_in_tank[-1] + water_pumped.iloc[i] - water_demand[i]
            if new_water_level < 0:
                water_deficit.append(abs(new_water_level))
                constrained_water_level = 0
            else:
                constrained_water_level = min(new_water_level, tank_capacity)
                water_deficit.append(0)
            water_in_tank.append(constrained_water_level)

        water_in_tank = water_in_tank
        water_deficit = water_deficit

        # return clpsp(water_deficit, water_demand)[-1]
        return lpsp_total(water_deficit, water_demand)
    
    def appraise_system(number_solar_panels: int, storage_factor: float, *args, **kwargs) -> float:
        cost = COST_POWER_INVERTER + COST_PUMP
        tank_capacity = 24 * storage_factor * WATER_DEMAND_HOURLY 
        cost += tank_capacity * COST_PER_M3_TANK
        cost += number_solar_panels * COST_PER_PANEL
        return cost/400

    losses = []
    costs = []
    for config in tqdm(hyperparams):
        config_loss = evaluate_sys(**config)
        config_cost = appraise_system(**config)
        logging.info(f"Configuration has loss {config_loss} and cost {config_cost}")
        losses.append(config_loss)
        costs.append(config_cost)
        

    
    min_loss_index = np.array(losses).argmin()
    logging.info(f"Best configuration: {min_loss_index} with loss {losses[min_loss_index]}")

    xs, ys = np.meshgrid(storage_factor_test, number_solar_panels_test, sparse=False)
    plt.contourf(xs, ys, np.array(losses).reshape(xs.shape))
    cb = plt.colorbar()
    cb.set_label("Loss (LPSP)")
    plt.xlabel("Storage Factor")
    plt.ylabel("Number of solar panels")
    plt.title("")
    plt.savefig("outputs/tradeoff_hourly.png")
    plt.clf()

    plt.contourf(xs, ys, np.array(costs).reshape(xs.shape))
    cb = plt.colorbar()
    cb.set_label("Cost per community member (USD)")
    plt.xlabel("Storage Factor")
    plt.ylabel("Number of solar panels")
    plt.title("")
    plt.savefig("outputs/cost_hourly.png")
    plt.clf()

    losses = np.array(losses)
    costs = np.array(costs)

    # Identify Pareto front
    pareto_indices = []
    for i in range(len(losses)):
        if not any((losses[j] <= losses[i] and costs[j] < costs[i]) or 
                  (losses[j] < losses[i] and costs[j] <= costs[i]) 
                  for j in range(len(losses)) if j != i):
            pareto_indices.append(i)

    pareto_losses = losses[pareto_indices]
    pareto_costs = costs[pareto_indices]

    # Sort Pareto front for plotting
    sorted_indices = np.argsort(pareto_costs)
    pareto_losses = pareto_losses[sorted_indices]
    pareto_costs = pareto_costs[sorted_indices]

    # Plot Pareto front
    plt.figure(figsize=(10, 6))
    plt.scatter(costs, losses, alpha=0.5, label="All configurations")
    plt.plot(pareto_costs, pareto_losses, color="red", marker="o", label="Pareto front")
    plt.xlabel("Cost per community member (USD)")
    plt.ylabel("Loss of Power Supply Probability (LPSP)")
    plt.title("")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/pareto_front_cost_vs_performance.png")
    plt.show()




    



