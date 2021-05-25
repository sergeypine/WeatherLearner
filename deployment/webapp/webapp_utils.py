import datetime
import logging
import sys
import pytz
from datetime import timedelta
import numpy
import pandas
from flask import current_app

sys.path.insert(1, '../')
import config
import libcommons.libcommons


def get_forecast_df():
    data_store = libcommons.libcommons.DataStore()
    predictions = data_store.predictions_load()
    if predictions is None:
        get_logger().warning("No weather predictions found")
        return None

    now_datetime = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE)).replace(tzinfo=None)
    future_predictions = predictions[predictions['DATE'] > now_datetime]
    if len(future_predictions) < 1:
        get_logger().warning("No future weather predictions found")
        return None

    # Predictions are sorted by DATE, then LOOK_AHEAD.
    # Keeping *first* dup results in considering the prediction with the smallest lookeahead (also the most precise)
    future_predictions = future_predictions.drop_duplicates(subset=['DATE', 'VAR'], keep='first')

    return future_predictions[['DATE', 'VAR', 'PREDICTION']]  # Drop LOOK_AHEAD: it is irrelevant


def get_current_conditions_df():
    data_store = libcommons.libcommons.DataStore()
    readings = data_store.readings_load(config.Config.TARGET_LOCATION)

    if readings is None or len(readings) < 1:
        get_logger().warning("No Weather Readings for Target Location Found")
        return None

    now_datetime = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE)).replace(tzinfo=None)
    readings = readings[readings['DATE'] < now_datetime]  # we need to do this due to padding to 24h
    last_reading = readings.iloc[-1]

    if (now_datetime - last_reading['DATE']).total_seconds() / 3600 > config.Config.WEBAPP_MAX_READING_DELAY_HOURS:
        get_logger().warning("No Recent Weather Readings for Target Location, last available is for {}".
                                format(last_reading['DATE']))
        return None

    return last_reading[['DATE', 'Temp', 'WindSpeed', '_is_clear', '_is_precip']]


def format_forecast(predictions_df):
    return predictions_df


# TODO - this is for testing not to break w.r.t. Flask Context. Explore a cleaner solution
def get_logger():
    try:
        return current_app.logger
    except RuntimeError:
        return logging
