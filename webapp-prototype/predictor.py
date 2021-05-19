import datetime
import sys
from datetime import timedelta

import numpy
import pandas
from flask import current_app
from tensorflow import keras
import pandas as pd
import numpy as np
import forecast_commons
import json
import itertools

numpy.set_printoptions(threshold=sys.maxsize)


def generate_predictions():
    predictions = init_predictions()
    base_timestamp = predictions['Timestamp'][0]  # blegh

    for prediction_target in current_app.config['ALL_PREDICTION_TARGETS']:
        current_app.logger.info("Generating prediction for var = {}, lookahead = {}hr".format(
            prediction_target.var, prediction_target.lookahead))

        # Get a Dataframe with data for locations and feature vars that are relevant to this prediction target
        featureset = forecast_commons.build_feature_set(prediction_target, "readings/{}.csv")

        # Normalize Data
        featureset = forecast_commons.normalize_data(featureset, prediction_target,
                                                     *load_mean_and_std(prediction_target))

        # Load the right model for this prediction
        model = load_target_model(prediction_target)

        # Generate a tensor that can be fed into that model to make prediction
        model_tensor = make_tensor(featureset, prediction_target)

        # Perform the prediction and record the result
        prediction = predict(prediction_target, model_tensor, model)

        # record result
        add_to_predictions(predictions,
                           base_timestamp + timedelta(hours=prediction_target.lookahead),
                           prediction_target.lookahead,
                           prediction_target.var,
                           prediction)

    predictions = pd.DataFrame.from_dict(predictions)
    predictions = predictions.sort_values(by=['Timestamp', 'Var'])
    current_app.logger.info(predictions)

    predictions.to_csv(forecast_commons.get_latest_predictions_file())


def load_target_model(prediction_target):
    model = keras.models.load_model(forecast_commons.get_model_file(prediction_target))
    current_app.logger.info("Loaded Model for var = {}, lookahead = {}hr".format(
        prediction_target.var, prediction_target.lookahead))
    current_app.logger.info(model.summary())
    return model


def make_tensor(featureset, prediction_target):
    # Remove all rows except for the ones needed to make prediction (last lookback hours rows)
    featureset = featureset.tail(current_app.config['PREDICTION_TARGET_LOOKBACKS'][prediction_target])

    # Add dummy rows up to lookahead time (WindowGenerator class needs to have rows for the forecast timeframe)
    for _ in itertools.repeat(None, prediction_target.lookahead - current_app.config['PREDICTED_VARIABLE_AHI'][
        prediction_target.var]):
        featureset = featureset.append(pd.Series(), ignore_index=True)

    agg_interval = 2 * current_app.config['PREDICTED_VARIABLE_AHI'][prediction_target.var] + 1
    wg = forecast_commons.WindowGenerator(
        input_width=current_app.config['PREDICTION_TARGET_LOOKBACKS'][prediction_target],
        label_width=agg_interval,
        shift=prediction_target.lookahead - current_app.config['PREDICTED_VARIABLE_AHI'][prediction_target.var],
        label_columns=[prediction_target.var],
        train_df=featureset,
        test_df=None,
        val_df=None)

    t = wg.train.__iter__().get_next()[0]
    return t


def predict(prediction_target, model_tensor, model):
    predictions = None
    agg_rule = current_app.config['PREDICTED_VARIABLE_AGG_RULES'][prediction_target.var]
    if agg_rule in ['ALL', 'ANY']:
        predictions = (model.predict(model_tensor, verbose=1) > 0.5).astype("int32").flatten()
    else:
        predictions = model.predict(model_tensor).flatten()

    if agg_rule == 'ANY':
        return 1 if sum(predictions) > 0 else 0
    elif agg_rule == 'ALL':
        return 1 if sum(predictions) == len(predictions) else 0
    else:
        return np.mean(predictions)


def load_mean_and_std(prediction_target):
    norm_file_name = forecast_commons.get_normalization_file(prediction_target)
    with open(norm_file_name) as json_file:
        mean_std_dict = json.load(json_file)
        return mean_std_dict['MEAN'], mean_std_dict['STD']


def init_predictions():
    predictions = {
        'Timestamp': [],
        'Lookahead': [],
        'Var': [],
        'Val': []
    }

    df = forecast_commons.load_location_readings(current_app.config['TARGET_LOCATION'], "readings/{}.csv")
    df = df.tail(1)
    timestamp = pd.to_datetime(df['DATE'].item())
    for var in current_app.config['PREDICTED_VARIABLE_AGG_RULES']:
        add_to_predictions(predictions, timestamp, 0, var, df[var].item())

    return predictions


def add_to_predictions(predictions, timestamp, lookahead, var, val):
    predictions['Timestamp'].append(timestamp)
    predictions['Lookahead'].append(lookahead)
    predictions['Var'].append(var)
    predictions['Val'].append(val)
