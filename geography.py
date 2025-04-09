
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
import geopandas as gpd
import csv

AFRICA_COUNTRIES = [
                "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
                "Cabo Verde", "Cameroon", "Central African Rep.", "Chad", "Comoros",
                "Dem. Rep. Congo", "Congo", "Côte d'Ivoire", "Djibouti", "Egypt",
                "Eq. Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana",
                "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya",
                "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
                "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Principe",
                "Senegal", "Seychelles", "Sierra Leone", "Somalia", "Somaliland",
                "South Africa", "S. Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
                "Zambia", "Zimbabwe", "W. Sahara"
            ]


def heatmap_style_map(ax, title, extent=None):
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=14)
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')


def plot_heatmap(lon_lat_pairs: list[tuple[float]], values: np.ndarray, output_file: str, legend: str, hue_style: str = "blue"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    if hue_style == "blue":
        cmap = cm.Blues
    elif hue_style == "red":
        cmap = cm.Reds
    elif hue_style == "orange":
        cmap = cm.Oranges
    elif hue_style == "green":
        cmap = cm.Greens
    else:
        raise ValueError("Invalid hue_style. Choose either 'blue', 'red', 'green', or 'orange'.")

    lon_within_mask = [coord[0] for coord in lon_lat_pairs]
    lat_within_mask = [coord[1] for coord in lon_lat_pairs]
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    colors = cmap(norm(values))

    buffer = 5
    min_lon, max_lon = min(lon_within_mask) - buffer, max(lon_within_mask) + buffer
    min_lat, max_lat = min(lat_within_mask) - buffer, max(lat_within_mask) + buffer
    ax.scatter(lon_within_mask, lat_within_mask, color=colors, s=20, transform=ccrs.PlateCarree())
    heatmap_style_map(ax, "", extent=[min_lon, max_lon, min_lat, max_lat])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02]) 
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(legend)

    plt.tight_layout(rect=[0, 0.15, 1, 1])  
    plt.savefig(output_file, dpi=300)
    plt.clf()

def plot_heatmap_binary(lon_lat_pairs: list[tuple[float, float]], 
                          bool_values, 
                          output_file: str, 
                          legend_true: str = "True", 
                          legend_false: str = "False", 
                          color_true: str = "blue", 
                          color_false: str = "red"):
    """
    Plots a binary heatmap on a map projection using given longitude-latitude pairs and a corresponding 
    array of boolean values. Points with True values are plotted in color_true and points with 
    False values in color_false. A legend is added to indicate the mapping.
    
    Parameters:
        lon_lat_pairs (list of tuple[float, float]): A list of (lon, lat) coordinate pairs.
        bool_values (list or np.ndarray): A boolean array (or list) with True/False values corresponding
                                          to each coordinate pair.
        output_file (str): The path to save the output plot.
        legend_true (str): Label for points where the value is True. Default is "True".
        legend_false (str): Label for points where the value is False. Default is "False".
        color_true (str): Color for points with True values. Default is "blue".
        color_false (str): Color for points with False values. Default is "red".
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Basic check that coordinate and boolean data have matching lengths
    if len(lon_lat_pairs) != len(bool_values):
        raise ValueError("Length of lon_lat_pairs must equal the length of bool_values.")
    
    # Separate longitude and latitude values
    lon_list = [coord[0] for coord in lon_lat_pairs]
    lat_list = [coord[1] for coord in lon_lat_pairs]
    
    # Map each boolean value to a color
    colors = [color_true if val else color_false for val in bool_values]
    
    # Define the buffer and extent for the map view
    buffer = 5
    min_lon, max_lon = min(lon_list) - buffer, max(lon_list) + buffer
    min_lat, max_lat = min(lat_list) - buffer, max(lat_list) + buffer
    
    # Create the figure and axis with Cartopy projection
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot the points colored by their boolean value
    ax.scatter(lon_list, lat_list, color=colors, s=20, transform=ccrs.PlateCarree())
    
    # Use the heatmap styling for the map (this function was defined elsewhere)
    # You can simply add features like land, ocean, borders, etc.
    def heatmap_style_map(ax, title, extent=None):
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=14)
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        # Adding grid lines with labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl.top_labels = gl.right_labels = False

    # Apply the styling to the plot
    heatmap_style_map(ax, "", extent=[min_lon, max_lon, min_lat, max_lat])
    
    # Create proxy artists for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=legend_true, markerfacecolor=color_true, markersize=8),
        Line2D([0], [0], marker='o', color='w', label=legend_false, markerfacecolor=color_false, markersize=8)
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
    
    # Save and clear the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.clf()

def plot_heatmap_category(lon_lat_pairs: list[tuple[float, float]], 
                          category_values, 
                          output_file: str, 
                          categories_order: list[str],
                          legend_labels: list[str],
                          colors: list[str]):
    """
    Plots a categorical heatmap on a map projection using given longitude-latitude pairs and corresponding
    category values.
    
    Parameters:
        lon_lat_pairs (list of tuple[float, float]): A list of (lon, lat) coordinate pairs.
        category_values (list or np.ndarray): A categorical array of identifiers corresponding to each coordinate.
        output_file (str): The path to save the output plot.
        categories_order (list of str): The list of expected category identifiers (e.g., ["SPWP", "Diesel", "Equivalent"]).
        legend_labels (list of str): Legend texts corresponding to each category (e.g., ["SPWP system is cost-effective", 
                                     "Diesel system is cost-effective", "Equivalent cost-effectiveness"]).
        colors (list of str): Colors corresponding to each category, in the same order as categories_order.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Validate input lengths.
    if len(lon_lat_pairs) != len(category_values):
         raise ValueError("Length of lon_lat_pairs must equal the length of category_values.")
    if not (len(categories_order) == len(legend_labels) == len(colors)):
         raise ValueError("categories_order, legend_labels, and colors must have the same length.")
    
    # Create a mapping from category identifiers to colors.
    mapping = dict(zip(categories_order, colors))
    
    # Verify that each category value is present in the expected categories.
    for cat in category_values:
        if cat not in categories_order:
             raise ValueError(f"Category '{cat}' not found in categories_order.")
    
    # Map each category value to its corresponding color.
    point_colors = [mapping[cat] for cat in category_values]
    
    # Unpack the longitude and latitude components from the coordinate pairs.
    lon_list = [coord[0] for coord in lon_lat_pairs]
    lat_list = [coord[1] for coord in lon_lat_pairs]
    
    # Determine map extent with a simple buffer.
    buffer = 5
    min_lon, max_lon = min(lon_list) - buffer, max(lon_list) + buffer
    min_lat, max_lat = min(lat_list) - buffer, max(lat_list) + buffer

    # Create the map figure using a Cartopy PlateCarree projection.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.scatter(lon_list, lat_list, color=point_colors, s=20, transform=ccrs.PlateCarree())

    # Add some base map features.
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Create legend handles for each category.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=leg, markerfacecolor=col, markersize=8)
        for leg, col in zip(legend_labels, colors)
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.clf()


def mask_lon_lat(lon: np.ndarray, lat: np.ndarray, country_name: str | None = None, continent: str | None = None, plot: bool = True) -> list[tuple[float, float]]:
    import regionmask

    countries = regionmask.defined_regions.natural_earth_v5_1_2.countries_50

    if country_name:
        country_index = countries.map_keys(country_name)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        mask = countries.mask(lon_grid, lat_grid)
        country_mask = mask == country_index
        lon_within_mask = lon_grid[country_mask]
        lat_within_mask = lat_grid[country_mask]
    elif continent == "Africa":
        countries_gdf = countries.to_geodataframe()
        africa_gdf = countries_gdf[countries_gdf["names"].isin(AFRICA_COUNTRIES)]
        africa_gdf.loc[africa_gdf["names"] == "Somaliland", "abbrevs"] = "SML" # Sierra Leone and Somaliland both have the same abbreviation as SL
        africa_gdf.loc[africa_gdf["names"] == "Seychelles", "abbrevs"] = "SYC" # Seychelles and Sierra Leone both have the same abbreviation as SRB
        africa_region = regionmask.Regions.from_geodataframe(africa_gdf, name="Africa")
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        mask = africa_region.mask(lon_grid, lat_grid)
        # For a union mask, gridpoints in Africa are non-NaN.
        country_mask = ~np.isnan(mask)
        lon_within_mask = lon_grid[country_mask]
        lat_within_mask = lat_grid[country_mask]
    elif country_name and continent:
        raise ValueError("You must provide either a country_name or continent, not both.")
    elif not country_name and not continent:
        raise ValueError("You must provide either a country_name or continent.")
    else: 
        raise NotImplementedError("This combination of country_name and continent is not supported.")


    coordinates_masked = []
    for lon,lat in zip(lon_within_mask, lat_within_mask):
        coordinates_masked.append((lon,lat))

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax1 = axes[0]
        ax1.scatter(lon_grid, lat_grid, color="lightgray", s=1, transform=ccrs.PlateCarree(), label="All Points")
        ax1.scatter(lon_within_mask, lat_within_mask, color="blue", s=5, transform=ccrs.PlateCarree(), label=f"Points in {country_name}")
        heatmap_style_map(ax1, "Global View")
        ax1.legend(loc="lower left", fontsize=10)

        ax2 = axes[1]
        buffer = 5 
        min_lon, max_lon = min(lon_within_mask) - buffer, max(lon_within_mask) + buffer
        min_lat, max_lat = min(lat_within_mask) - buffer, max(lat_within_mask) + buffer
        ax2.scatter(lon_grid, lat_grid, color="lightgray", s=1, transform=ccrs.PlateCarree(), label="All Points")
        ax2.scatter(lon_within_mask, lat_within_mask, color="blue", s=5, transform=ccrs.PlateCarree(), label=f"Points in {country_name}")
        heatmap_style_map(ax2, f"Zoomed View of {country_name}", extent=[min_lon, max_lon, min_lat, max_lat])
        ax2.legend(loc="lower left", fontsize=10)

        plt.tight_layout()
        plt.savefig("outputs/lon_lat_mask_with_map.png", dpi=300)
        plt.clf()

    return coordinates_masked


def mask_landmass(lon: np.ndarray, lat: np.ndarray):
    import regionmask
    countries = regionmask.defined_regions.natural_earth_v5_1_2.land_50
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = countries.mask(lon_grid, lat_grid)
    land_mask = ~np.isnan(mask)    # shape (len(lat), len(lon))

    lon_within_mask = lon_grid[land_mask]
    lat_within_mask = lat_grid[land_mask]
    coordinates_masked = list(zip(lon_within_mask, lat_within_mask))
    return coordinates_masked, land_mask

