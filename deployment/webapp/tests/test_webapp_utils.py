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
             'Temp': [90, 91],
             'WindSpeed': [1, 2],
             '_is_clear': [0, 1],
             '_is_precip': [1, 0],
             'Pressure': [29.2, 29.3],
             'WindGust': [20, 21],
             'DewPoint': [90, 91],
             'Humidity': [81, 82],
             '_wind_dir_sin': [0.7, 0.8],
             '_wind_dir_cos': [0.2, 0.3]}))

        current_conditions = webapp_utils.get_current_conditions_df()
        assert current_conditions is None

        # Case 3: Current readings for Target Location
        accepted_ts = now_datetime - datetime.timedelta(hours=config.Config.WEBAPP_MAX_READING_DELAY_HOURS - 1)
        prev_ts = accepted_ts - datetime.timedelta(hours=1)
        next_ts = accepted_ts + datetime.timedelta(hours=1)

        ds.readings_append(config.Config.TARGET_LOCATION, pd.DataFrame.from_dict(
            {'DATE': [prev_ts.replace(tzinfo=None), accepted_ts.replace(tzinfo=None), next_ts.replace(tzinfo=None)],
             'Temp': [100, 101, 101],
             'WindSpeed': [10, 11, 11],
             '_is_clear': [0, 1, 1],
             '_is_precip': [1, 0, 0],
             'Pressure': [29.2, 29.3, 29.3],
             'WindGust': [20, 21, 21],
             'DewPoint': [90, 91, 91],
             'Humidity': [81, 82, 82],
             '_wind_dir_sin': [0.7, 0.8, 0.8],
             '_wind_dir_cos': [0.2, 0.3, 0.3]}))

        cd = webapp_utils.get_current_conditions_df()
        assert len(cd) == 4
        assert cd.iloc[0]['DATE'] == pd.Timestamp(accepted_ts.replace(tzinfo=None))
        assert cd.iloc[0]['VAR'] == 'Temp'
        assert cd.iloc[0]['PREDICTION'] == 101
        assert cd.iloc[1]['DATE'] == pd.Timestamp(accepted_ts.replace(tzinfo=None))
        assert cd.iloc[1]['VAR'] == 'WindSpeed'
        assert cd.iloc[1]['PREDICTION'] == 11
        assert cd.iloc[2]['DATE'] == pd.Timestamp(accepted_ts.replace(tzinfo=None))
        assert cd.iloc[2]['VAR'] == '_is_clear'
        assert cd.iloc[2]['PREDICTION'] == 1
        assert cd.iloc[3]['DATE'] == pd.Timestamp(accepted_ts.replace(tzinfo=None))
        assert cd.iloc[3]['VAR'] == '_is_precip'
        assert cd.iloc[3]['PREDICTION'] == 0

        # next_ds entry imitates a padded entry and should be ignored


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
        ds.predictions_append(future_ts1, config.PredictionTarget('Temp', 6), 70)  # Duplicate: should be kept
        ds.predictions_append(future_ts1, config.PredictionTarget('Temp', 12), 75)  # Duplicate: should be dropped
        ds.predictions_append(future_ts1, config.PredictionTarget('_is_clear', 6), 1)  # Duplicate: should be kept
        ds.predictions_append(future_ts1, config.PredictionTarget('_is_clear', 24), 0)  # Duplicate: should be dropped
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


def test_format_forecast():
    date1 = '2020-03-01 04:00:00'
    date2 = '2020-03-01 10:00:00'
    date3 = '2020-03-01 16:00:00'
    date4 = '2020-03-01 22:00:00'
    predictions_df = pd.DataFrame.from_dict(
        {'DATE': [date1, date1, date1, date1,
                  date2, date2, date2, date2,
                  date3, date3, date3, date3,
                  date4, date4, date4, date4],
         'VAR': ['Temp', 'WindSpeed', '_is_clear', '_is_precip',
                 'Temp', 'WindSpeed', '_is_clear', '_is_precip',
                 'Temp', 'WindSpeed', '_is_clear', '_is_precip',
                 'Temp', 'WindSpeed', '_is_clear', '_is_precip'],
         'PREDICTION': [25.2, 10.3, 0, 1,  # Precipitation (snow)
                        40.1, 5.2, 0, 1,  # Precipitation (rain)
                        45.4, 2.0, 0, 0,  # Cloudy
                        35.3, 8.4, 1, 0]  # Clear
         }
    )

    formatted_df = webapp_utils.format_forecast(predictions_df)
    print(formatted_df)
    assert len(formatted_df) == 4
    assert formatted_df.iloc[0]['Temperature'] == '25 F'
    assert formatted_df.iloc[1]['Temperature'] == '40 F'
    assert formatted_df.iloc[2]['Temperature'] == '45 F'
    assert formatted_df.iloc[3]['Temperature'] == '35 F'

    assert formatted_df.iloc[0]['Wind'] == '10 mph'
    assert formatted_df.iloc[1]['Wind'] == '5 mph'
    assert formatted_df.iloc[2]['Wind'] == '2 mph'
    assert formatted_df.iloc[3]['Wind'] == '8 mph'

    assert formatted_df.iloc[0]['Conditions'] == 'Snow'
    assert formatted_df.iloc[1]['Conditions'] == 'Rain'
    assert formatted_df.iloc[2]['Conditions'] == 'Cloudy'
    assert formatted_df.iloc[3]['Conditions'] == 'Clear'
