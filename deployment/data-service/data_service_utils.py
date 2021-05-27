import datetime
import sys
import time

import pandas as pd
import pytz

import libcommons.libcommons

sys.path.insert(1, '../')
import config


def get_dates_for_forecast(conf: config.Config, forecast_time=None):
    if forecast_time is None:
        now_datetime = datetime.datetime.now(pytz.timezone(conf.TARGET_TIMEZONE))
    else:
        now_datetime = forecast_time
    lookback_datetime = now_datetime - datetime.timedelta(hours=conf.MAX_LOOK_BACK_HOURS)

    dates = []
    current_date = lookback_datetime.date()
    while current_date <= now_datetime.date():
        dates.append(datetime.datetime.combine(current_date, datetime.datetime.min.time()))
        current_date = current_date + datetime.timedelta(days=1)

    return dates


def get_date_locations_to_retrieve(conf: config.Config):
    date_locations = {}
    # Backfill starts with yesterday: today is handled by the forecast generation logic
    now_datetime = datetime.datetime.now(pytz.timezone(conf.TARGET_TIMEZONE))
    current_datetime = (now_datetime - datetime.timedelta(days=1)).replace(
        tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    end_datetime = current_datetime - datetime.timedelta(days=conf.DATA_SERVICE_BACKFILL_INTERVAL_DAYS)

    ds = libcommons.libcommons.DataStore()

    # Keep going back day by day until we either reach the BATCH_SIZE or reach end of LOOKBACK_WINDOW
    while len(list(date_locations.keys())) < conf.DATA_SERVICE_BACKFILL_BATCH_SIZE and current_datetime > end_datetime:
        for location in list(conf.LOCATION_CODES.keys()):
            readings = ds.readings_load(location)  # TODO - consider caching location readings
            current_date_readings = readings[readings['DATE'].dt.date == current_datetime.date()]
            if len(current_date_readings) < 24:
                if current_datetime not in date_locations:
                    date_locations[current_datetime] = []
                date_locations[current_datetime].append(location)

        current_datetime = current_datetime - datetime.timedelta(days=1)

    return date_locations


def get_missing_timestamp_prediction_targets(conf: config.Config):
    timestamp_prediction_targets = {}
    ds = libcommons.libcommons.DataStore()

    now_datetime = datetime.datetime.now(pytz.timezone(conf.TARGET_TIMEZONE)).replace(
        tzinfo=None, second=0, microsecond=0)
    dates_with_missing_readings = list(get_date_locations_to_retrieve(conf).keys())
    dates_with_missing_readings.sort()

    # If dates are missing, start with the TOMORROW relative to the LATEST missing date
    if len(dates_with_missing_readings) > 0:
        start_date_time = dates_with_missing_readings[-1] + datetime.timedelta(days=1)
    # If no dates are missing, start with BACKFILL_INTERVAL days before now
    else:
        start_date_time = now_datetime - datetime.timedelta(days=conf.DATA_SERVICE_BACKFILL_INTERVAL_DAYS)

    # But further advance starting time with MAX_LOOKBACK_TIME (so we are sure there is enough past readings to predict)
    start_date_time = start_date_time + datetime.timedelta(hours=conf.MAX_LOOK_BACK_HOURS)

    # Look for missing predictions from the Starting Time calculated above to Now
    predictions = ds.predictions_load()
    current_datetime = start_date_time
    while current_datetime < now_datetime:
        for prediction_target in conf.ALL_PREDICTION_TARGETS:
            forecast_date_time = current_datetime + datetime.timedelta(hours=prediction_target.lookahead)
            matching_predictions = predictions[(predictions['VAR'] == prediction_target.var) &
                                               (predictions['LOOK_AHEAD'] == prediction_target.lookahead) &
                                               (predictions['DATE'] == forecast_date_time)]
            if len(matching_predictions) < 1:
                if current_datetime not in timestamp_prediction_targets:
                    timestamp_prediction_targets[current_datetime] = []
                timestamp_prediction_targets[current_datetime].append(prediction_target)

        current_datetime = current_datetime + datetime.timedelta(hours=1)

    return timestamp_prediction_targets
