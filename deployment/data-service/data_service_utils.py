import datetime
import sys
import time
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
        dates.append(current_date)
        current_date = current_date + datetime.timedelta(days=1)

    return dates
