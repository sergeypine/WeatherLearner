import datetime
import sys
import time

import pandas as pd
import numpy as np
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


def get_actual_weather_history(conf: config.Config):
    readings = libcommons.libcommons.DataStore().readings_load(conf.TARGET_LOCATION)
    actual_weather_history = pd.DataFrame()

    for var_name in conf.PREDICTED_VARIABLE_AHI:
        actual_weather_history[var_name] = \
            list(readings[[var_name]].rolling(window=2 * conf.PREDICTED_VARIABLE_AHI[var_name] + 1,
                                            min_periods=1, center=True).mean().values)

        if conf.PREDICTED_VARIABLE_AGG_RULES[var_name] == 'ALL':
            actual_weather_history[var_name] = np.where(actual_weather_history[var_name] == 1, 1, 0)
        elif conf.PREDICTED_VARIABLE_AGG_RULES[var_name] == 'ANY':
            actual_weather_history[var_name] = np.where(actual_weather_history[var_name] == 0, 0, 1)

        actual_weather_history[var_name] = actual_weather_history[var_name].astype(int)
    actual_weather_history['DATE'] = readings['DATE']
    return actual_weather_history


def get_prediction_audit_df(target_var):
    data_store = libcommons.libcommons.DataStore()
    actual_weather_history = data_store.actual_weather_history_load()
    predictions = data_store.predictions_load()

    if actual_weather_history is None or predictions is None:
        return None

    # Trim weather history and predictions to only include data for target var
    predictions = predictions[predictions['VAR'] == target_var]
    actual_weather_history = actual_weather_history[['DATE', target_var]]

    # Merge history and predictions
    merge_df = pd.DataFrame.merge(actual_weather_history, predictions, on=['DATE'])
    merge_df = merge_df.rename(columns={target_var: 'Actual'})

    # Create empty result DF with the right columns
    audit_columns = ['DATE', 'Actual']
    lookaheads = map(lambda pt: pt.lookahead,
                     filter(lambda pt: pt.var == target_var, config.Config.ALL_PREDICTION_TARGETS))
    lookahead_cols = map(lambda l: "+{}h".format(l), lookaheads)
    audit_columns.extend(lookahead_cols)
    audit_df = pd.DataFrame(columns=audit_columns)

    # One row per DATE
    audit_df['DATE'] = merge_df['DATE'].unique()

    # Populate result DF row by row
    for index, row in merge_df.iterrows():
        ts = row['DATE']
        audit_df.loc[audit_df['DATE'] == ts, 'Actual'] = row['Actual']

        la = row['LOOK_AHEAD']
        audit_df.loc[audit_df['DATE'] == ts, "+{}h".format(la)] = row['PREDICTION']

    return audit_df