import datetime

import pandas as pd
import pytest
import sys
import os
import shutil
import datetime
import pytz

sys.path.insert(1, '../')
import data_service_utils
import config
import libcommons
import copy


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
    """Verify all missing dates are picked up and LOOKBACK_WINDOW is respected"""
    # TODO - consider testing that BATCH_SIZE is  respected too
    try:
        old_ds_path = config.Config.DATA_STORE_BASE_DIR
        config.Config.DATA_STORE_BASE_DIR = "tests/test_datastore"
        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

        conf = copy.deepcopy(config.Config)
        conf.LOCATION_CODES = {'Loc1': 'CODE1', 'Loc2': 'CODE2'}
        conf.DATA_SERVICE_BACKFILL_INTERVAL_DAYS = 3

        ds = libcommons.libcommons.DataStore()

        now_datetime = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE)).replace(
            tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        date1 = now_datetime - datetime.timedelta(days=1)
        date2 = now_datetime - datetime.timedelta(days=2)
        date3 = now_datetime - datetime.timedelta(days=3)
        date4 = now_datetime - datetime.timedelta(days=4)

        # The Situation:
        #   - Date1 has data for all locations
        #   - Date2 has no data at all
        #   - Date3 missing data for one location
        #   - Date4 missing data for one location but is outside backfill interval
        date1_loc1_data = make_day_of_data(date1)
        date1_loc2_data = make_day_of_data(date1)
        date3_loc2_data = make_day_of_data(date3)
        date4_loc1_data = make_day_of_data(date4)
        ds.readings_append('Loc1', date1_loc1_data)
        ds.readings_append('Loc2', date1_loc2_data)
        ds.readings_append('Loc2', date3_loc2_data)
        ds.readings_append('loc1', date4_loc1_data)

        date_locations = data_service_utils.get_date_locations_to_retrieve(conf)
        returned_dates = list(date_locations.keys())
        assert len(returned_dates) == 2
        assert date2 in returned_dates
        assert date3 in returned_dates
        assert len(date_locations[date2]) == 2
        assert len(date_locations[date3]) == 1
        assert "Loc1" in date_locations[date2]
        assert "Loc2" in date_locations[date2]
        assert "Loc1" in date_locations[date3]
    finally:
        shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)
        config.Config.DATA_STORE_BASE_DIR = old_ds_path


def test_get_missing_timestamp_prediction_targets():
    try:
        """Verify logic for finding missing predictions finds all of them provided necessary readings exist"""
        old_ds_path = config.Config.DATA_STORE_BASE_DIR
        config.Config.DATA_STORE_BASE_DIR = "tests/test_datastore"
        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

        conf = copy.deepcopy(config.Config)

        pt_temp_6h = config.PredictionTarget('Temp', 6)
        pt_temp_12h = config.PredictionTarget('Temp', 12)
        pt_windspeed_6h = config.PredictionTarget('WindSpeed', 6)
        pt_windspeed_12h = config.PredictionTarget('WindSpeed', 12)
        conf.ALL_PREDICTION_TARGETS = [
            pt_temp_6h, pt_temp_12h,
            pt_windspeed_6h, pt_windspeed_12h
        ]

        conf.DATA_SERVICE_BACKFILL_INTERVAL_DAYS = 5
        conf.LOCATION_CODES = {conf.TARGET_LOCATION: 'CODE1'}
        conf.MAX_LOOK_BACK_HOURS = 24

        ds = libcommons.libcommons.DataStore()

        now_datetime = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE)).replace(
            tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        date1 = now_datetime - datetime.timedelta(days=1)
        date2 = now_datetime - datetime.timedelta(days=2)
        date3 = now_datetime - datetime.timedelta(days=3)
        date4 = now_datetime - datetime.timedelta(days=4)

        # Situation:
        #   - Readings present for Date1, Date2 and Date4 (Date3 is missing!)
        #   - All predictions for today are filled in
        #   - Some Predictions are missing for Date1, Date2, Date3 and Date4
        #   - ... However, only the missing predictions for Date1 should be flagged
        #         (because there are no readings to predict for Date2)
        ds.readings_append(conf.TARGET_LOCATION, make_day_of_data(date1))
        ds.readings_append(conf.TARGET_LOCATION, make_day_of_data(date2))
        ds.readings_append(conf.TARGET_LOCATION, make_day_of_data(date4))

        all_predictions = []
        for _date in [date1, date2, date3, date4]:
            predictions = make_day_of_predictions(_date, pt_temp_6h)
            del predictions[0]
            all_predictions.extend(predictions)
            # --
            predictions = make_day_of_predictions(_date, pt_temp_12h)
            del predictions[23]
            all_predictions.extend(predictions)
            # --
            predictions = make_day_of_predictions(_date, pt_windspeed_6h)
            del predictions[1:3]
            all_predictions.extend(predictions)
            # --
            predictions = make_day_of_predictions(_date, pt_windspeed_12h)
            del predictions[1]
            del predictions[2]
            all_predictions.extend(predictions)
            # --
        all_predictions.extend(make_day_of_predictions(now_datetime, pt_temp_6h))
        all_predictions.extend(make_day_of_predictions(now_datetime, pt_temp_12h))
        all_predictions.extend(make_day_of_predictions(now_datetime, pt_windspeed_6h))
        all_predictions.extend(make_day_of_predictions(now_datetime, pt_windspeed_12h))
        for prediction in all_predictions:
            ds.predictions_append(*prediction)

        missing_timestamp_prediction_targets = data_service_utils.get_missing_timestamp_prediction_targets(conf)
        missing_timestamps = list(missing_timestamp_prediction_targets.keys())
        assert len(missing_timestamps) == 5
        assert len(missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 0, 0)]) == 1
        assert missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 0, 0)][0] == pt_temp_6h

        assert len(missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 23, 0)]) == 1
        assert missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 23, 0)][0] == pt_temp_12h

        assert len(missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 1, 0)]) == 2
        assert missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 1, 0)][0] == pt_windspeed_6h
        assert missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 1, 0)][1] == pt_windspeed_12h

        assert len(missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 2, 0)]) == 1
        assert missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 2, 0)][0] == pt_windspeed_6h

        assert len(missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 3, 0)]) == 1
        assert missing_timestamp_prediction_targets[datetime.datetime(date1.year, date1.month, date1.day, 3, 0)][0] == pt_windspeed_12h

    finally:
        shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)
        config.Config.DATA_STORE_BASE_DIR = old_ds_path


def test_get_actual_weather_history():
    """Test reading aggregation that mimicks model training"""
    try:
        old_ds_path = config.Config.DATA_STORE_BASE_DIR
        config.Config.DATA_STORE_BASE_DIR = "tests/test_datastore"
        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

        conf = copy.deepcopy(config.Config)
        conf.PREDICTED_VARIABLE_AHI = {
            '_is_precip': 3,
            '_is_clear': 3,
            'Temp': 1,
            'WindSpeed': 2,
        }
        conf.PREDICTED_VARIABLE_AGG_RULES = {
            '_is_precip': 'ANY',
            '_is_clear': 'ALL',
            'Temp': 'AVG',
            'WindSpeed': 'AVG',
        }

        date_rows = ['2018-05-05 10:00:00', '2018-05-05 11:00:00', '2018-05-05 12:00:00', '2018-05-05 13:00:00',
                     '2018-05-05 14:00:00', '2018-05-05 15:00:00', '2018-05-05 16:00:00', '2018-05-05 17:00:00',
                     '2018-05-05 18:00:00']
        temp_rows =     [50, 51, 52, 53, 54, 55, 55, 54, 53]
        expected_temp = [50, 51, 52, 53, 54, 54, 54, 54, 53]

        wind_rows =     [10, 11, 12, 13, 12, 11, 10, 9, 8]
        expected_wind = [11, 11, 11, 11, 11, 11, 10, 9, 9]

        is_clear_rows =     [0, 1, 1, 1, 1, 1, 1, 1, 0]
        expected_is_clear = [0, 0, 0, 0, 1, 0, 0, 0, 0]

        is_precip_rows =    [1, 0, 0, 0, 0, 0, 0, 0, 1]
        expected_is_precip =[1, 1, 1, 1, 0, 1, 1, 1, 1]

        readings = pd.DataFrame.from_dict({
            'DATE': date_rows,
            'Temp': temp_rows,
            'WindSpeed': wind_rows,
            '_is_clear': is_clear_rows,
            '_is_precip': is_precip_rows
        })
        libcommons.libcommons.DataStore().readings_append(conf.TARGET_LOCATION, readings)

        actual_weather_history = data_service_utils.get_actual_weather_history(conf)

        assert 'DATE' in actual_weather_history.columns.values
        assert list(actual_weather_history['Temp'].values) == expected_temp
        assert list(actual_weather_history['WindSpeed'].values) == expected_wind
        assert list(actual_weather_history['_is_clear'].values) == expected_is_clear
        assert list(actual_weather_history['_is_precip'].values) == expected_is_precip

    finally:
        shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)
        config.Config.DATA_STORE_BASE_DIR = old_ds_path


#  -----------------------------------------------------------------
#  --- Helper routines
#  -----------------------------------------------------------------
def make_day_of_data(day_date):
    day_date = datetime.datetime.combine(day_date.date(), datetime.datetime.min.time())
    dates = []
    data = []

    for hr in range(0, 24):
        dates.append(day_date + datetime.timedelta(hours=hr))
        data.append(100 * hr)

    return pd.DataFrame.from_dict({"DATE": dates, "DATA": data})


def make_day_of_predictions(day_date, prediction_target):
    # prediction_time, prediction_target, predicted_val
    day_date = datetime.datetime.combine(day_date.date(), datetime.datetime.min.time())
    predictions = []
    for hr in range(0, 24):
        # Param Order: prediction_time, prediction_target, predicted_val
        predictions.append([day_date + datetime.timedelta(hours=hr + prediction_target.lookahead),
                            prediction_target,
                            1000000])
    return predictions


def clear_data_store():
    if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
        shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)
