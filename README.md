# Weather Predictor by Sergey Numerov

## What this is:

ML-driven application that forecasts weather in Chicago IL in the next 24h. 
The underlying models are trained on official NOAA hourly data sets for Chicago
and a number of nearby locations in the Midwest, such as Indianapolis. 
Provides a simple web UI to display data and model information.

This is my Capstone Project for the University of San Diego Machine Learning Bootcamp program.

## Detailed Information:

[High Level Project Description](https://github.com/sergeypine/WeatherLearner/blob/main/doc/project_proposal.pdf)

[Technical Architecture](https://github.com/sergeypine/WeatherLearner/blob/main/doc/Weather_Predictor_Architecture.pdf)

[ML Modeling Info](https://github.com/sergeypine/WeatherLearner/blob/main/model-prototyping/model_experimenting.ipynb)

[Dataset](https://github.com/sergeypine/WeatherLearner/tree/main/raw-data)

[Data Exploration](https://github.com/sergeypine/WeatherLearner/blob/main/data-processing/data-wrangling.ipynb)


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
- you may not want to always run all stages of `build.py`. If so, add command line arguments from the following set to hand pick stages: 
   - `train`
   - `build`
   - `deploy-local`
   - `deploy-remote`
- file `deployment/config.py` configures nearly everything about the application. You may want to take a look at:
    - `PREDICTION_TARGET_FEATURES` and `PREDICTION_TARGET_LOCATIONS` determine Features and Geographic Locations used to train each of the 16 models
    - `DATA_SERVICE_BACKFILL_INTERVAL_DAYS` determines how far back historic data is pulled to compile Predictions Audit (history)
    - `DATA_SERVICE_FORECAST_INTERVAL_MINUTES` and `DATA_SERVICE_BACKFILL_INTERVAL_MINUTES` determine how often Forecast and Prediction Audit get refreshed
    - `PREDICTION_TARGET_MODEL_TYPES` sets the _kind_ of each model (`LINEAR`, `NN` (1 layer), `DNN` (2 layer) and `CNN` (Convolutional))
  
## Deploy to Cloud

The following is for deploying to AWS. It should not be too difficult to modify `build.py` to deploy to another Cloud Provider

### Prerequisites

- AWS Account 
- locally installed and configured AWS CLI (run `aws configure`)
- `credentials_etc.json` file (copy `credentials_etc_template.json` and set values accordingly)
- The following resources exist in the AWS account (you'll need to create them manually :( ):  
  - `weather-predictor` repository  in AWS ECR that is initially empty
  - ECS Cluster and Task `weather-predictor` assigned to it that is associated with a single EC2 instance of type `t2.medium`
  - Task is set to run the Docker image in the ECR repository  
  - Task in Network Mode `Host`
  - Task Port is 5000
  - Service `weather-predictor` is associated with the task  
  - EC2 instance _Inbound Rules_ must allow traffic on ports 80 (HTTP), 5000 (Custom TCP) and 22 (SSH)

### Command Line

Assuming you previously deployed locally (as per instructions above):

```
python3 build.py deploy-remote
```

Alternatively, you may specify the relevant command line arguments to `build.py` as discussed above. 
For instance, if you've changed UI but don't want models retrained, run:

```
python3 build.py build deploy-remote
```

After the above completes, the application Web UI should be available via the following URL:

http://ec5-55-555-555-555.<your_aws_region>.compute.amazonaws.com:5000/forecast

where `ec5-55-555-555-555` needs to be substituted with your EC2 instance's Public IPv4 DNS and `your_aws_region` for the region in which you made the ECS task

