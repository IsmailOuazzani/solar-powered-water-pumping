from pathlib import Path
import pandas as pd
from geography import mask_lon_lat, heatmap_style_map, plot_heatmap, plot_heatmap_binary, plot_heatmap_category
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from scipy.stats import binned_statistic
from cartopy import crs as ccrs
from statsmodels.nonparametric.smoothers_lowess import lowess
from pv_system import make_pv_system

INPUT_PATH = Path("outputs") / "thesis_input"
OUTPUT_PATH = Path("outputs") / "thesis_output"


C_1YEAR_CONSTANT = INPUT_PATH / "best_configs_1year_constant.csv"
C_1YEAR_DIURNAL = INPUT_PATH / "best_configs_diurnal_.csv"
C_1YEAR_CONSTANT_LOW_LOSS = INPUT_PATH / "best_configs_1year_constant_0.0005.csv"
C_10YEAR_CONSTANT = INPUT_PATH / "best_config_10year_constant.csv"

def open_csv(file_path):
    df = pd.read_csv(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.rename(columns={"longitude": "lat", "latitude": "lon"}, inplace=True)

    df.rename(columns={
        "number_solar_panels": "Number of Solar Panels",
        "storage_factor": "Storage Factor",
        "lat": "Latitude (°)",
        "lon": "Longitude (°)",
        "cost": "Cost per User ($USD)",
        "loss": "Loss",
        "avg_tank_volume": "Average Tank Volume (m³)",
        "var_tank_volume": "Variance in Tank Volume (m³)",
        "shortage_hours": "Shortage Hours",
    }, inplace=True)
    return df



import matplotlib.pyplot as plt

def analyze_and_plot_datasets(
    dataset1,
    dataset2,
    dataset1_name,
    dataset2_name,
    x_vars,
    y_vars,
    x_labels,
    y_labels
):
    """
    Generates scatter plots comparing variable relationships between two datasets,
    computes the absolute and percentage differences for cost per user, number of solar panels,
    and storage factor increase, prints a summary statistics table to the CLI, and plots histograms
    for these differences.

    Parameters:
        dataset1: First dataset (e.g., baseline/constant demand)
        dataset2: Second dataset (e.g., diurnal demand)
        dataset1_name: String label for the first dataset (e.g., "constant demand")
        dataset2_name: String label for the second dataset (e.g., "diurnal demand")
        x_vars: List of column names for x-axis variables.
        y_vars: List of column names for y-axis variables.
        x_labels: List of labels (strings) for the x-axes.
        y_labels: List of labels (strings) for the y-axes.

    The function assumes that both datasets contain the following columns:
    "Cost per User ($USD)", "Storage Factor", and "Number of Solar Panels".
    """
    # Auto-generate a base name from the dataset names (replace spaces with underscores)
    base_name = f"{dataset1_name.replace(' ', '_')}_{dataset2_name.replace(' ', '_')}"
    
    # ---------------------------
    # Scatter Plot Generation
    # ---------------------------
    n = len(x_vars)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=300)
    # Ensure axes is always iterable (works even if n == 1)
    if n == 1:
        axes = [axes]

    for i, (x_var, y_var, x_label, y_label) in enumerate(zip(x_vars, y_vars, x_labels, y_labels)):
        ax = axes[i]
        # Plot dataset1 points
        ax.scatter(
            dataset1[x_var],
            dataset1[y_var],
            s=20,
            alpha=0.7,
            color="#4D4D4D",  # dark grey
            label=dataset1_name
        )
        # Plot dataset2 points
        ax.scatter(
            dataset2[x_var],
            dataset2[y_var],
            s=20,
            alpha=0.7,
            color="#7B3294",  # purple
            label=dataset2_name
        )
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend(fontsize=10)
        
    plt.tight_layout()
    scatter_filename = f"scatter_{base_name}.png"
    plt.savefig(scatter_filename, dpi=300)
    plt.show()
    
    # ---------------------------
    # Compute Differences
    # ---------------------------
    # Cost per User Differences (absolute and percentage)
    cost_abs = dataset2["Cost per User ($USD)"] - dataset1["Cost per User ($USD)"]
    cost_percent = (cost_abs / dataset1["Cost per User ($USD)"]) * 100
    
    # Storage Factor Differences (absolute and percentage)
    storage_abs = dataset2["Storage Factor"] - dataset1["Storage Factor"]
    storage_percent = ((dataset2["Storage Factor"] - dataset1["Storage Factor"]) 
                       / dataset1["Storage Factor"]) * 100
    
    # Number of Solar Panels Differences (absolute and percentage)
    solar_abs = dataset2["Number of Solar Panels"] - dataset1["Number of Solar Panels"]
    solar_percent = (solar_abs / dataset1["Number of Solar Panels"]) * 100

    # ---------------------------
    # Print Summary Statistics
    # ---------------------------
    print(f"\nSummary Statistics {dataset1_name} vs {dataset2_name}:\n")
    
    # Cost per User
    print("Cost per User ($USD):")
    print(f"  Absolute difference: Mean = {cost_abs.mean():.2f}, Std = {cost_abs.std():.2f}")
    print(f"  Percentage difference: Mean = {cost_percent.mean():.2f}%, Std = {cost_percent.std():.2f}%\n")
    
    # Storage Factor
    print("Storage Factor:")
    print(f"  Absolute difference: Mean = {storage_abs.mean():.2f}, Std = {storage_abs.std():.2f}")
    print(f"  Percentage difference: Mean = {storage_percent.mean():.2f}%, Std = {storage_percent.std():.2f}%\n")
    
    # Number of Solar Panels
    print("Number of Solar Panels:")
    print(f"  Absolute difference: Mean = {solar_abs.mean():.2f}, Std = {solar_abs.std():.2f}")
    print(f"  Percentage difference: Mean = {solar_percent.mean():.2f}%, Std = {solar_percent.std():.2f}%\n")
    
    # ---------------------------
    # Histogram Plotting
    # ---------------------------
    # Define a helper function to create and save histograms.
    def save_histogram(data, xlabel, filename):
        plt.figure(figsize=(10, 6), dpi=300)
        plt.hist(data, bins=20, color="#4D4D4D", alpha=0.7, edgecolor='black')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    # Cost per User histograms
    cost_abs_filename = OUTPUT_PATH/f"hist_cost_abs_{base_name}.png"
    save_histogram(cost_abs,
                   "Absolute Cost Increase ($USD)",
                   cost_abs_filename)
    
    cost_percent_filename = OUTPUT_PATH/f"hist_cost_percent_{base_name}.png"
    save_histogram(cost_percent,
                   "Percentage Cost Increase (%)",
                   cost_percent_filename)
    
    # Storage Factor histograms
    storage_abs_filename = OUTPUT_PATH/f"hist_storage_abs_{base_name}.png"
    save_histogram(storage_abs,
                   "Absolute Storage Factor Increase",
                   storage_abs_filename)
    
    storage_percent_filename = OUTPUT_PATH/f"hist_storage_percent_{base_name}.png"
    save_histogram(storage_percent,
                   "Storage Factor Increase (%)",
                   storage_percent_filename)
    
    # Number of Solar Panels histograms
    solar_abs_filename = OUTPUT_PATH/f"hist_solar_abs_{base_name}.png"
    save_histogram(solar_abs,
                   "Increase in Number of Solar Panels",
                   solar_abs_filename)
    
    solar_percent_filename = OUTPUT_PATH/f"hist_solar_percent_{base_name}.png"
    save_histogram(solar_percent,
                   "Percentage Increase in Number of Solar Panels (%)",
                   solar_percent_filename)




####################################### Simulation Parameters #######################################
import pvlib
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['AstroPower_AP_1206___1998_'] 
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

# Print all module and inverter parameters
print("Module parameters:")
for key, value in module.items():
    print(f"{key}: {value}")
print("\nInverter parameters:")
for key, value in inverter.items():
    print(f"{key}: {value}")


###################################### Baseline Optimization ######################################\
baseline = open_csv(C_1YEAR_CONSTANT)

# Correlation Matrix
baseline_corr =  baseline.drop(columns=["pattern"]).corr()
plt.figure(figsize=(20,20), dpi=300) 
sns.heatmap(baseline_corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig(OUTPUT_PATH / "baseline_correlation_matrix.png", dpi=300)

# Big variable plot
variables = baseline.columns.tolist()
n = len(variables)
fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
for i, var_y in enumerate(variables):
    for j, var_x in enumerate(variables):
        ax = axes[i, j]
        # On the diagonal, plot a histogram
        if i == j:
            ax.hist(baseline[var_x], bins=20,color="#7B3294", edgecolor='black')
            ax.set_title(f"{var_x} histogram", fontsize=10)
            # For histograms, label the x-axis with the variable and the y-axis with "Frequency"
            ax.set_xlabel(var_x, fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
        else:
            # Off-diagonal: scatter plot of var_x vs var_y
            ax.scatter(baseline[var_x], baseline[var_y], s=20, color="#7B3294", alpha=0.7)
            # For scatter plots, label the axes with their respective variable names
            ax.set_xlabel(var_x, fontsize=8)
            ax.set_ylabel(var_y, fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "baseline_pairwise_plots.png", dpi=200)

# Plot heatmap of cost
lon_lat_pairs = baseline[["Longitude (°)", "Latitude (°)"]]
lon_lat_pairs = lon_lat_pairs.apply(lambda x: (x[1], x[0]), axis=1)
costs = baseline["Cost per User ($USD)"]
plot_heatmap(
    lon_lat_pairs=lon_lat_pairs.values,
    values=costs.values,
    output_file=OUTPUT_PATH / "baseline_cost.png",
    hue_style="green",
    legend="Cost per User ($USD)",
)

# Where it is worth it
cost_diesel_per_user = 72
costs = baseline["Cost per User ($USD)"]
def categorize_cost(cost, threshold):
    tolerance = 2
    if cost < threshold - tolerance:
         return "SPWP"
    elif cost > threshold + tolerance:
         return "Diesel"
    else:
         return "Equivalent"
categories = costs.apply(lambda x: categorize_cost(x, cost_diesel_per_user))
lon_lat_pairs = baseline[["Longitude (°)", "Latitude (°)"]]
lon_lat_pairs = lon_lat_pairs.apply(lambda x: (x[1], x[0]), axis=1)
categories_order = ["SPWP", "Diesel", "Equivalent"]
legend_labels = [
    "SPWP system is cost-effective",
    "Diesel system is cost-effective",
    "Equivalent"
]
colors = ["green", "red", "yellow"]
plot_heatmap_category(
    lon_lat_pairs=lon_lat_pairs.values,
    category_values=categories.values,
    output_file=OUTPUT_PATH / "baseline_worth_it.png",
    categories_order=categories_order,
    legend_labels=legend_labels,
    colors=colors
)

######################## Diurnal Demand ########################
baseline = open_csv(C_1YEAR_CONSTANT)
diurnal = open_csv(C_1YEAR_DIURNAL)



# Longitude latitiudecost x3 then longitude latitude longitude latitude
x_labels = x_vars = [
    "Longitude (°)", "Latitude (°)", "Cost per User ($USD)",
    "Latitude (°)", "Longitude (°)", "Cost per User ($USD)",
    "Longitude (°)", "Latitude (°)", "Longitude (°)", "Latitude (°)"
]
# var tnk volume x 3 avg tank volume x3 shortage hours x2 cost x2
y_labels = y_vars = [
    "Variance in Tank Volume (m³)", "Variance in Tank Volume (m³)", "Variance in Tank Volume (m³)",
    "Average Tank Volume (m³)", "Average Tank Volume (m³)", "Average Tank Volume (m³)",
    "Shortage Hours", "Shortage Hours", "Cost per User ($USD)", "Cost per User ($USD)"
]

analyze_and_plot_datasets(
    dataset1=baseline,
    dataset2=diurnal,
    dataset1_name="constant demand",
    dataset2_name="diurnal demand",
    x_vars=x_vars,
    y_vars=y_vars,
    x_labels=x_labels,
    y_labels=y_labels,
)


######### 10 year constant demand #########
baseline = open_csv(C_1YEAR_CONSTANT)
long = open_csv(C_10YEAR_CONSTANT)

analyze_and_plot_datasets(
    dataset1=baseline,
    dataset2=long,
    dataset1_name="1 year",
    dataset2_name="10 years",
    x_vars=x_vars,
    y_vars=y_vars,
    x_labels=x_labels,
    y_labels=y_labels,
)