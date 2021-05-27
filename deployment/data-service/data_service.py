import datetime
import sys
import time
import pytz
import schedule
import logging

sys.path.insert(1, '../')
import config
import reading_retriever
import predictor
import data_service_utils


def generate_forecast_job():
    logging.info("!!! START Forecast Job")

    forecast_dates = data_service_utils.get_dates_for_forecast(conf)

    # (1) Get Weather Readings for all locations and for the last 2 days
    for location in conf.LOCATION_CODES:
        for forecast_date in forecast_dates:
            logging.info("Retrieving data for location {}, date {}".format(location, forecast_date))
            reading_retriever.retrieve_for_date_and_location(forecast_date, location)

    # (2) Generate Forecast
    for prediction_target in conf.ALL_PREDICTION_TARGETS:
        logging.info("Forecasting for Prediction Target {}".format(prediction_target))

        # Predict based on the current time
        base_time = datetime.datetime.now(pytz.timezone(conf.TARGET_TIMEZONE))
        base_time = datetime.datetime(base_time.year, base_time.month, base_time.day, base_time.hour, 0, 0)

        predictor.predict_for_target_and_base_time(prediction_target, base_time)

    logging.info("!!! END Forecast Job")


def backfill_readings_job():
    logging.info("!!! START Backfill Readings Job")

    # date->location dictionary of readings we are missing
    missing_dates_locations = data_service_utils.get_date_locations_to_retrieve(conf)
    missing_dates = list(missing_dates_locations.keys())
    missing_dates.sort()

    # retrieve missing data
    if len(missing_dates) > 0:
        logging.info("Missing weather readings for the following dates will be backfilled: {}".format(missing_dates))
        for missing_date in missing_dates:
            for location in missing_dates_locations[missing_date]:
                logging.info("Backfilling missing readings, location {}, date {}".format(location, missing_date))
                reading_retriever.retrieve_for_date_and_location(missing_date, location)
    else:
        logging.info("No Weather Readings are missing")

    logging.info("!!! END Backfill Readings Job")


def backfill_predictions_job():
    logging.info("!!! START Backfill Predictions Job")

    missing_timestamp_prediction_targets = \
        data_service_utils.get_missing_timestamp_prediction_targets(conf)
    missing_timestamps = list(missing_timestamp_prediction_targets.keys())
    missing_timestamps.sort()

    if len(missing_timestamps) > 0:
        logging.info("Recent predictions for the following TS will be backfilled: {}".format(missing_timestamps))
        for missing_ts in missing_timestamps:
            prediction_targets = missing_timestamp_prediction_targets[missing_ts]
            for prediction_target in prediction_targets:
                logging.info("Backfilling recent prediction for Target {}, base Timestamp {}".format(
                    prediction_target, missing_ts))
                predictor.predict_for_target_and_base_time(prediction_target, missing_ts)
    else:
        logging.info("No recent predictions are missing")

    logging.info("!!! END Backfill Predictions Job")


def main():
    # Pre-execute the jobs
    generate_forecast_job()
    backfill_readings_job()
    backfill_predictions_job()

    # Keep running the jobs on a schedule
    schedule.every(conf.DATA_SERVICE_FORECAST_INTERVAL_MINUTES).minutes.do(generate_forecast_job)
    schedule.every(conf.DATA_SERVICE_BACKFILL_INTERVAL_MINUTES).minutes.do(backfill_readings_job)
    schedule.every(conf.DATA_SERVICE_BACKFILL_INTERVAL_MINUTES).minutes.do(backfill_predictions_job)
    while 1:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    conf = config.Config
    logging.basicConfig(filename=conf.DATA_SERVICE_LOG_FILE,
                        format=conf.DATA_SERVICE_LOG_FORMAT,
                        level=conf.DATA_SERVICE_LOG_LEVEL)

    reading_retriever = reading_retriever.ReadingRetriever()
    predictor = predictor.Predictor()
    main()
