import datetime
import pytz

import pytest
import sys
import os
import shutil
import pandas as pd

sys.path.insert(1, '../')
import webapp_utils
import config
import libcommons


def test_get_current_conditions_df():
    try:
        old_ds_path = config.Config.DATA_STORE_BASE_DIR
        config.Config.DATA_STORE_BASE_DIR = "tests/test_data"

        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

        ds = libcommons.libcommons.DataStore()

        # Case 1: No readings for Target Location
        assert webapp_utils.get_current_conditions_df() is None

        # Case 2: Old readings for Target Location
        now_datetime = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE))

        accepted_ts = now_datetime - datetime.timedelta(hours=config.Config.WEBAPP_MAX_READING_DELAY_HOURS + 1)
        prev_ts = accepted_ts - datetime.timedelta(hours=1)

        ds.readings_append(config.Config.TARGET_LOCATION, pd.DataFrame.from_dict(
            {'DATE': [prev_ts.replace(tzinfo=None), accepted_ts.replace(tzinfo=None)],
             'Temp': [100, 101],
             'WindSpeed': [10, 11],
             '_is_clear': [0, 1],
             '_is_precip': [1, 0]}))

        current_conditions = webapp_utils.get_current_conditions_df()
        assert current_conditions is None

        # Case 3: Current readings for Target Location
        accepted_ts = now_datetime - datetime.timedelta(hours=config.Config.WEBAPP_MAX_READING_DELAY_HOURS - 1)
        prev_ts = accepted_ts - datetime.timedelta(hours=1)

        ds.readings_append(config.Config.TARGET_LOCATION, pd.DataFrame.from_dict(
            {'DATE': [prev_ts.replace(tzinfo=None), accepted_ts.replace(tzinfo=None)],
             'Temp': [100, 101],
             'WindSpeed': [10, 11],
             '_is_clear': [0, 1],
             '_is_precip': [1, 0]}))

        current_conditions = webapp_utils.get_current_conditions_df()
        assert current_conditions['DATE'] == pd.Timestamp(accepted_ts).replace(tzinfo=None)
        assert current_conditions['Temp'] == 101
        assert current_conditions['WindSpeed'] == 11
        assert current_conditions['_is_clear'] == 1
        assert current_conditions['_is_precip'] == 0

    finally:
        config.Config.DATA_STORE_BASE_DIR = old_ds_path


def test_get_forecast_df():
    try:
        old_ds_path = config.Config.DATA_STORE_BASE_DIR
        config.Config.DATA_STORE_BASE_DIR = "tests/test_data"

        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

        ds = libcommons.libcommons.DataStore()

        # Case 1: no predictions at all
        assert webapp_utils.get_forecast_df() is None

        # Case 2: predictions are old
        now_datetime = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE))

        # Case 3: predictions are future
        future_ts1 = (now_datetime + datetime.timedelta(hours=6)).replace(tzinfo=None)
        future_ts2 = future_ts1 + datetime.timedelta(hours=12)
        ds.predictions_append(future_ts1, config.PredictionTarget('Temp', 6), 70) # Duplicate: should be kept
        ds.predictions_append(future_ts1, config.PredictionTarget('Temp', 12), 75) # Duplicate: should be dropped
        ds.predictions_append(future_ts1, config.PredictionTarget('_is_clear', 6), 1) # Duplicate: should be kept
        ds.predictions_append(future_ts1, config.PredictionTarget('_is_clear', 24), 0) # Duplicate: should be dropped
        ds.predictions_append(future_ts2, config.PredictionTarget('Temp', 6), 80)

        forecast = webapp_utils.get_forecast_df()
        assert len(forecast) == 3
        assert forecast.iloc[0]['DATE'] == future_ts1
        assert forecast.iloc[1]['DATE'] == future_ts1
        assert forecast.iloc[2]['DATE'] == future_ts2
        assert forecast.iloc[0]['PREDICTION'] == 70
        assert forecast.iloc[1]['PREDICTION'] == 1
        assert forecast.iloc[2]['PREDICTION'] == 80
        assert forecast.iloc[0]['VAR'] == 'Temp'
        assert forecast.iloc[1]['VAR'] == '_is_clear'
        assert forecast.iloc[2]['VAR'] == 'Temp'


    finally:
        config.Config.DATA_STORE_BASE_DIR = old_ds_path
