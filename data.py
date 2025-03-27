"""Utilities to import datasets"""
import xarray as xr
from pathlib import Path
import logging
from glob import glob
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


DATASETS = {
    "M2T1NXRAD_5-2023_only_SWGDN": "datasets/M2T1NXRAD_5-2023_only_SWGDN.nc4"
}

def extract_variables(data_folder: Path, variable: str, dataset_output_path: Path):
    """Extract a single variable from multiple NetCDF files and save it to a single NetCDF 
    file without compression for fast reading."""

    assert Path(data_folder).exists(), f"Data folder {data_folder} does not exist."
    dataset_name = f"{data_folder.as_posix().rstrip('/')}_only_{variable}" #TODO: simplify this logic
    dataset_file = dataset_output_path/f"{dataset_name}.nc4"

    nc_files = sorted(glob(f"{data_folder}/*.nc4"))
    logging.debug(f"Found {len(nc_files)} NetCDF files.")
    datasets = []
    for nc_file in tqdm(nc_files, desc="Processing files", unit="file"):
        logging.debug(f"Processing {nc_file}")
        ds = xr.open_dataset(nc_file)[variable].load()  # Load into memory
        datasets.append(ds)

    combined_dataset = xr.concat(datasets, dim="time")
    combined_dataset.to_netcdf(dataset_file, engine="netcdf4", format="NETCDF4", encoding={variable: {"zlib": False}})
    logging.info(f"Saved combined dataset to {dataset_file}")

def import_merra2_dataset(dataset_path: Path, variables: list[str]) -> xr.Dataset:
    logging.debug(f"Opening dataset at {dataset_path}...")
    ds = xr.open_dataset(dataset_path)[variables]
    logging.info(f"Successfuly opened dataset. Summary:\n{ds}") 
    return ds

    
if __name__ == "__main__":
    # extract_variables(
    #     data_folder=Path("/media/ismail/WDC/datasets/M2T1NXRAD_5-2023"),
    #     dataset_output_path=Path("/media/ismail/WDC/datasets"),
    #     variable="SWGDN"
    # )
    # TODO: turn this into argparse... or click?
    ...