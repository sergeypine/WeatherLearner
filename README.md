# Weather Predictor by Sergey Numerov

## What this is:

ML-driven application that forecasts weather in Chicago IL in the next 24h. 
The underlying models are trained on official NOAA hourly data sets for Chicago
and a number of nearby locations in the Midwest, such as Indianapolis. 
Provides a simple web UI to display data and model information.

This is my Capstone Project for the University of San Diego Machine Learning Bootcamp program.

## How to run it:

These instructions are for running the Docker container containing the application locally.

### Prerequisites
- Python3
- all dependencies as per `deployment/requirements.txt`  
- Docker
- ability to open port `5000` on `localhost` (exact port may be changed in `build.py`)

### Command Line

```
cd deployment
export PYTHONPATH=. pytest
python3 build.py
```

This will take a couple of hours to run. When done, application will be available at `http://localhost:5000/forecast`

If something goes wrong or you are just curious what the app is doing, run the following:

```
cd deployment
docker exec -it weather-predictor  /bin/bash
```
To look at the web application logs:
```
cd logs
tail -f  webapp.log
```

To look at the data and prediction service logs (this is where most of the real work is done):
```
cd logs
tail -f data_service.log
```

## Experiment

- `../raw-data` directory contains training datasets from NOAA (to be manually ordered from https://www.ncdc.noaa.gov/cdo-web/datatools/lcd). Each file must be in `.csv` format and the name must contain location with the following capitalization: `Des_Moines`. You can add/remove such files from that directory to train on different data 
- you may not want to always run all stages of `build.py`. If so, edit flags such as `do_train` found in the beginning of the script
- file `deployment/config.py` configures nearly everything about the application. You may want to take a look at:
    - `PREDICTION_TARGET_FEATURES` and `PREDICTION_TARGET_LOCATIONS` determine Features and Geographic Locations used to train each of the 16 models
    - `DATA_SERVICE_BACKFILL_INTERVAL_DAYS` determines how far back historic data is pulled to compile Predictions Audit (history)
    - `DATA_SERVICE_FORECAST_INTERVAL_MINUTES` and `DATA_SERVICE_BACKFILL_INTERVAL_MINUTES` determine how often Forecast and Prediction Audit get refreshed
    - `PREDICTION_TARGET_MODEL_TYPES` sets the _kind_ of each model (`LINEAR`, `NN` (1 layer), `DNN` (2 layer) and `CNN` (Convolutional))