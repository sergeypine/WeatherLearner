import datetime
import logging
import sys

import pandas as pd
import numpy as np
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

    # we need to do this due to padding to 24h
    readings = readings[readings['DATE'] < now_datetime]
    readings = readings.drop_duplicates(subset=['Temp', 'DewPoint', 'Humidity', 'WindSpeed', 'WindGust', 'Pressure',
                                                '_wind_dir_sin', '_wind_dir_cos'], keep='first')
    if readings is None or len(readings) < 1:
        get_logger().warning("No Recent Weather Readings for Target Location Found")
        return None

    last_reading = readings.iloc[-1]

    if (now_datetime - last_reading['DATE']).total_seconds() / 3600 > config.Config.WEBAPP_MAX_READING_DELAY_HOURS:
        get_logger().warning("No Recent Weather Readings for Target Location, last available is for {}".
                             format(last_reading['DATE']))
        return None

    last_reading =  last_reading[['DATE', 'Temp', 'WindSpeed', '_is_clear', '_is_precip']]
    current_conditions_df = pd.DataFrame(columns=['DATE', 'VAR', 'PREDICTION'])
    current_conditions_df = \
        current_conditions_df.append(pd.Series([last_reading['DATE'], 'Temp', last_reading['Temp']]
                                               , index=current_conditions_df.columns), ignore_index=True)
    current_conditions_df = \
        current_conditions_df.append(pd.Series([last_reading['DATE'], 'WindSpeed', last_reading['WindSpeed']]
                                               , index=current_conditions_df.columns), ignore_index=True)
    current_conditions_df = \
        current_conditions_df.append(pd.Series([last_reading['DATE'], '_is_clear', last_reading['_is_clear']]
                                               , index=current_conditions_df.columns), ignore_index=True)
    current_conditions_df = \
        current_conditions_df.append(pd.Series([last_reading['DATE'], '_is_precip', last_reading['_is_precip']]
                                               , index=current_conditions_df.columns), ignore_index=True)

    print(current_conditions_df)
    return current_conditions_df


def format_forecast(predictions_df: pd.DataFrame):
    formatted_df = pd.DataFrame(columns=['Timestamp', 'Temperature', 'Wind', 'Conditions'])

    # One row per date
    dates = predictions_df['DATE'].unique()
    formatted_df['Timestamp'] = dates

    for _date in dates:
        # NOTE - race condition here during the first run of backfill (since forecasts are missing for some vars)
        temp = (predictions_df[(predictions_df['DATE'] == _date) &
                               (predictions_df['VAR'] == 'Temp')])['PREDICTION'].values[0]
        formatted_df.loc[formatted_df['Timestamp'] == _date, 'Temperature'] = "{} F".format(int(temp))

        wind = (predictions_df[(predictions_df['DATE'] == _date) &
                               (predictions_df['VAR'] == 'WindSpeed')])['PREDICTION'].values[0]
        formatted_df.loc[formatted_df['Timestamp'] == _date, 'Wind'] = "{} mph".format(int(wind))

        _is_clear = (predictions_df[(predictions_df['DATE'] == _date) &
                               (predictions_df['VAR'] == '_is_clear')])['PREDICTION'].values[0]
        _is_precip = (predictions_df[(predictions_df['DATE'] == _date) &
                               (predictions_df['VAR'] == '_is_precip')])['PREDICTION'].values[0]
        formatted_df.loc[formatted_df['Timestamp'] == _date, 'Conditions'] = get_conditions(temp, _is_clear, _is_precip)

    return formatted_df


def format_yesno_tbl(yesno_tbl):
    for col in yesno_tbl.columns:
        if col != 'DATE':
            yesno_tbl[col] = np.where(yesno_tbl[col] == 0, 'No', 'Yes')
    return yesno_tbl


def format_numeric_tbl(numeric_tbl):
    for col in numeric_tbl.columns:
        if col != 'DATE':
            numeric_tbl[col] = numeric_tbl[col].astype(int)

    return  numeric_tbl


def reduce_audit_granurality(audit_df):
    # TODO - use WEBAPP_HRS_INCLUDED_AUDIT instead of hardcoding the hours
    return audit_df[(audit_df['DATE'].dt.hour == 2) |
                    (audit_df['DATE'].dt.hour == 8) |
                    (audit_df['DATE'].dt.hour == 14) |
                    (audit_df['DATE'].dt.hour == 20)]


def get_conditions(temp, _is_clear, _is_precip):
    # No Precipitation
    if _is_clear == 1 and _is_precip == 0:
        return 'Clear'
    if _is_clear == 0 and _is_precip == 0:
        return 'Cloudy'

    # Precipitation (NOTE: figuring out type of precipitation from temperature is very approximate)
    if temp < 30:
        return 'Snow'
    if temp >= 40:
        return 'Rain'

    return 'Winter Precipitation'

# TODO - this is for testing not to break w.r.t. Flask Context. Explore a cleaner solution
def get_logger():
    try:
        return current_app.logger
    except RuntimeError:
        return logging
