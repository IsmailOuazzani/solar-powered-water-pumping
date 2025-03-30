
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

