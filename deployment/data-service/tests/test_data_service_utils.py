import datetime

import pytest
import sys

sys.path.insert(1, '../')
import data_service_utils
import config


def test_get_dates_for_forecast():
    """Test with different lookback times and forecast times"""
    # Default is lookback = 24h and forecast time is now
    conf = config.Config
    dates = data_service_utils.get_dates_for_forecast(conf)
    assert len(dates) == 2

    conf.MAX_LOOK_BACK_HOURS = 6
    forecast_time = datetime.datetime(2020, 10, 10, 5, 0, 0)
    dates = data_service_utils.get_dates_for_forecast(conf, forecast_time=forecast_time)
    assert len(dates) == 2
    assert dates[0] == datetime.datetime(2020, 10, 9)
    assert dates[1] == datetime.datetime(2020, 10, 10)

    forecast_time = datetime.datetime(2020, 10, 10, 7, 0, 0)
    dates = data_service_utils.get_dates_for_forecast(conf, forecast_time=forecast_time)
    assert dates[0] == datetime.datetime(2020, 10, 10)
    assert len(dates) == 1


def test_get_date_locations_to_retrieve():
    pass


def test_missing_dates_to_prediction_timestamps():
    pass


def test_get_missing_timestamp_prediction_targets():
    pass
