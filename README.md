# Solar Powered Water Pumping

## Set up
Create a virtual environment and install dependencies:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datasets
For a guide MERRA-2, please consult the [MERRA-2: File Specification](https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf).

### Download datasets
Follow GES DISC's [Data Access Instructions](https://disc.gsfc.nasa.gov/information/documents?title=Data%20Access).

Generate an EarthData Token at `https://urs.earthdata.nasa.gov/users/<your-username>/user_tokens`. And export it ans an env variable:
```
export EARTHDATA_TOKEN=<YOUR TOKEN>
```

Find the dataset you are interested at: [https://disc.gsfc.nasa.gov/datasets](https://disc.gsfc.nasa.gov/datasets).

Click "Subset/Get Data", select "Get Original Files" (default option), then "Get Data". Download the Links list into your folder of choice, write the absolute path in the `list_path` variable of `download_dataset.sh` and choose the output directory.

Run the dataset download script:
```
sh download_dataset.sh
```


