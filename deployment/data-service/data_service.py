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
    logging.info("Start Forecast Job")

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

    logging.info("End Forecast Job")


def update_prediction_audit_job():
    logging.info("Start Backfill Job")
    logging.info("End Backfill Job")


def main():
    # Pre-execute the jobs
    generate_forecast_job()
    update_prediction_audit_job()

    # Keep running the jobs on a schedule
    schedule.every(conf.DATA_SERVICE_FORECAST_INTERVAL_MINUTES).minutes.do(generate_forecast_job)
    schedule.every(conf.DATA_SERVICE_BACKFILL_INTERVAL_MINUTES).minutes.do(update_prediction_audit_job)
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
