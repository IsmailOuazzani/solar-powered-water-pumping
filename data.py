"""Utilities to import datasets"""
import xarray as xr
from pathlib import Path
import logging
import gc
from glob import glob
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASETS = {
    "M2T1NXRAD_5-2023_only_SWGDN": "datasets/M2T1NXRAD_5-2023_only_SWGDN.nc4"
}

def extract_variables(data_folder: Path, variable: str, dataset_output_path: Path, num_chunks: int = 5):
    """Extract a single variable from multiple NetCDF files and save it to a single NetCDF file.
    
    This version divides the files into `num_chunks` groups. Each group's data (only the 
    specified variable) is saved to a temporary file in a cache folder, and then the temporary 
    files are concatenated along the 'time' dimension into the final output file.
    """
    # Ensure the data folder exists
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    
    # Define the dataset name and final output file path
    dataset_name = f"{data_folder.name}_only_{variable}"
    dataset_file = dataset_output_path / f"{dataset_name}.nc4"
    
    # Create a cache folder for temporary files
    cache_folder = data_folder.parent / "cache"
    cache_folder.mkdir(exist_ok=True)
    logger.info(f"Cache folder: {cache_folder}")  
    nc_files = sorted(glob(f"{data_folder}/*.nc4"))
    logging.debug(f"Found {len(nc_files)} NetCDF files.")
    
    # Divide the list of files into roughly equal chunks
    chunk_size = len(nc_files) // num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        if i == num_chunks - 1:
            # The last chunk takes any remaining files
            chunk = nc_files[start:]
        else:
            chunk = nc_files[start:start + chunk_size]
            start += chunk_size
        chunks.append(chunk)
    
    temp_files = []
    # Process each chunk: extract the variable, concatenate along time, and save to cache
    for i, chunk in enumerate(chunks):
        datasets = []
        for nc_file in tqdm(chunk, desc=f"Processing chunk {i+1}", unit="file"):
            logging.debug(f"Processing {nc_file}")
            with xr.open_dataset(nc_file) as ds:
                data = ds[variable].load()  # load the data into memory
                datasets.append(data)
        # Concatenate the datasets for this chunk
        combined_chunk = xr.concat(datasets, dim="time")
        temp_file = cache_folder / f"{dataset_name}_chunk_{i+1}.nc4"
        combined_chunk.to_netcdf(
            temp_file, 
            engine="netcdf4", 
            format="NETCDF4", 
            encoding={variable: {"zlib": False}}
        )
        logging.info(f"Saved chunk {i+1} to {temp_file}")
        temp_files.append(temp_file)

        # Explicitly delete intermediate variables and run garbage collection
        del datasets, combined_chunk
        gc.collect()
    
    # Combine the temporary chunk files into the final dataset
    combined_datasets = []
    for temp_file in temp_files:
        ds = xr.open_dataset(temp_file, engine="netcdf4")[variable]
        combined_datasets.append(ds)
    
    combined_dataset = xr.concat(combined_datasets, dim="time")
    combined_dataset.to_netcdf(
        dataset_file, 
        engine="netcdf4", 
        format="NETCDF4", 
        encoding={variable: {"zlib": False}}
    )
    logging.info(f"Saved combined dataset to {dataset_file}")

def import_merra2_dataset(dataset_path: Path, variables: list[str]) -> xr.Dataset:
    logging.debug(f"Opening dataset at {dataset_path}...")
    ds = xr.open_dataset(dataset_path)[variables]
    logging.info(f"Successfuly opened dataset. Summary:\n{ds}") 
    return ds

    
if __name__ == "__main__":
    extract_variables(
        data_folder=Path("/media/ismail/BIG/datasets/M2T1NXRAD_5-2015-2025_only_SWGDN"),
        dataset_output_path=Path("/home/ismail/code/solar-powered-water-pumping/datasets/"),
        variable="SWGDN"
    )
    # TODO: turn this into argparse... or click?
    ...
