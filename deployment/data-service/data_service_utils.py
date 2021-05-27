import datetime
import sys
import time

import pandas as pd
import pytz

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
    now_datetime = datetime.datetime.now(pytz.timezone(conf.TARGET_TIMEZONE))
    return {}


def missing_dates_to_prediction_timestamps(missing_dates):
    #  TODO - cut off timestamps within MAX_LOOKBACK
    return []


def get_missing_timestamp_prediction_targets(lookback_hours=24):
    return {}
