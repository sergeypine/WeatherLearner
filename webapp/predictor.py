import sys

import numpy
from flask import current_app
from tensorflow import keras
import pandas as pd
import window_generator
numpy.set_printoptions(threshold=sys.maxsize)

def generate_predictions():
    predictions = {}

    for prediction_target in current_app.config['ALL_PREDICTION_TARGETS']:
        current_app.logger.info("Generating prediction for var = {}, lookahead = {}hr".format(
            prediction_target.var, prediction_target.lookahead))

        # Get a Dataframe with data for locations and feature vars that are relevant to this prediction target
        featureset = build_feature_set(prediction_target)

        # TODO - Normalize it!

        # Load the right model for this prediction
        model = load_target_model(prediction_target)

        # Generate a tensor that can be fed into that model to make prediction
        model_tensor = make_tensor(featureset, prediction_target)

        # Perform the prediction and record the result
        prediction = predict(model_tensor, model)
        predictions[prediction_target] = prediction
        current_app.logger.info("Prediction for var = {}, lookahead = {}hr is **{}**".format(
            prediction_target.var, prediction_target.lookahead, prediction
        ))


def build_feature_set(prediction_target):
    target_df = load_location_readings(current_app.config['TARGET_LOCATION'])
    target_df = drop_unused_columns(target_df, prediction_target)
    merged_df = target_df
    suffix_no = 1

    # Merge adjacent location files one by one relying on DATA
    for adjacent_location in current_app.config['PREDICTION_TARGET_LOCATIONS'][prediction_target]:
        adjacent_df = load_location_readings(adjacent_location)

        # Take control of column name suffix in the dataset being merged in
        adjacent_df = adjacent_df.add_suffix(str(suffix_no))
        adjacent_df = adjacent_df.rename(columns={"DATE{}".format(suffix_no): 'DATE'})
        merged_df = pd.merge(merged_df, adjacent_df, on='DATE')
        suffix_no = suffix_no + 1

    # DATA column is of no use in the modelling stage
    merged_df = merged_df.drop(columns=['DATE'])
    return merged_df


def load_target_model(prediction_target):
    model = keras.models.load_model("pretrained/{}_{}h.h5".format(prediction_target.var, prediction_target.lookahead))
    current_app.logger.info("Loaded Model for var = {}, lookahead = {}hr".format(
        prediction_target.var, prediction_target.lookahead))
    current_app.logger.info(model.summary())
    return model


def make_tensor(featureset, prediction_target):
    agg_interval = 2 * current_app.config['PREDICTED_VARIABLE_AHI'][prediction_target.var] + 1
    wg = window_generator.WindowGenerator(
        input_width=current_app.config['PREDICTION_TARGET_LOOKBACKS'][prediction_target],
        label_width=agg_interval,
        shift=prediction_target.lookahead - current_app.config['PREDICTED_VARIABLE_AHI'][prediction_target.var],
        label_columns=[prediction_target.var],
        train_df=featureset,
        test_df=None,
        val_df=None)
    # TODO - WindowGenerator does do exactly what we need it to.
    for element in wg.train.as_numpy_iterator():
        current_app.logger.info("=============================")
        current_app.logger.info(element)


def predict(model_tensor, model):
    return None


def drop_unused_columns(df, prediction_target):
    features_to_use = current_app.config['PREDICTION_TARGET_FEATURES'][prediction_target]

    all_columns = features_to_use.copy()
    all_columns.append('DATE')
    all_columns.append(prediction_target.var)
    df = df[all_columns]

    return df


def load_location_readings(location):
    loc_readings_df = pd.read_csv("readings/{}.csv".format(location))
    return loc_readings_df
