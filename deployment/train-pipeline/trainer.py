import datetime
import logging
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import math
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error

import sys

sys.path.insert(1, '../')
import libcommons.libcommons
import config

MAX_EPOCHS = 100
evaluations = {
    'REGRESSION': [],
    'CLASSIFICATION': []
}


def create_model(prediction_target, model_to_use, preprocessed_readings_dir):
    conf = config.Config
    feature_set_builder = libcommons.libcommons.FeatureSetBuilder()
    # Load the Target Location and Adjacent Locations Datasets and drop the unused columns
    featureset = feature_set_builder.build_feature_set(prediction_target,
                                                       base_time=None,
                                                       readings_csv_directory=preprocessed_readings_dir,
                                                       keep_all_dates=True)
    # Train/validation Split: 70% train, 30% validate
    train_end_idx = int(len(featureset) * 0.70)
    from_ts = (featureset['DATE'].astype(str))[0]
    to_ts = featureset['DATE'].astype(str)[train_end_idx-1]

    # DATE is no longer of use to us
    featureset = featureset.drop(columns=['DATE'])

    # Do the splitting
    train_df = featureset[0: train_end_idx]
    val_df = featureset[train_end_idx:]

    # Normalize input data (NOTE: TF tutorial also scales the target variable)
    train_mean = train_df[feature_set_builder.get_columns_to_normalize(train_df, prediction_target)].mean()
    train_std = train_df[feature_set_builder.get_columns_to_normalize(train_df, prediction_target)].std()
    train_df = feature_set_builder.normalize_data(train_df, prediction_target, train_mean, train_std)
    val_df = feature_set_builder.normalize_data(val_df, prediction_target, train_mean, train_std)

    # Generate Timeseries Prediction Window on which TensorFlow will operate
    ahi_interval = conf.PREDICTED_VARIABLE_AHI[prediction_target.var]
    agg_interval = 2 * ahi_interval + 1
    wg = libcommons.libcommons.WindowGenerator(
        input_width=conf.PREDICTION_TARGET_LOOKBACKS[prediction_target],
        label_width=agg_interval,
        shift=prediction_target.lookahead - ahi_interval,
        label_columns=[prediction_target.var],
        train_df=train_df, val_df=val_df, test_df=None)

    # Build the desired model
    input_cnt = (len(conf.PREDICTION_TARGET_LOCATIONS[prediction_target]) + 1) * \
                (len(conf.PREDICTION_TARGET_FEATURES[prediction_target]) + 1) * \
                conf.PREDICTION_TARGET_LOOKBACKS[prediction_target]
    logging.info("Input Cnt = {}".format(input_cnt))
    is_binary = 'is' in prediction_target.var

    model = None
    if model_to_use == 'LINEAR':
        model = build_model_linear(is_binary, agg_interval, input_cnt)

    if model_to_use == 'NN':
        model = build_model_nn(is_binary, agg_interval, input_cnt)

    if model_to_use == 'DNN':
        model = build_model_dnn(is_binary, agg_interval, input_cnt)

    if model_to_use == 'CNN':
        model = build_model_cnn(is_binary, agg_interval, input_cnt)

    # Configure early stopping so we don't learn the training data too well at the expense of test/validation data (
    # overfit) (use different criteria for Classification and Regression)
    if is_binary:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_recall', mode="max", patience=20, min_delta=0.0002,
                                                       restore_best_weights=True)
    else:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', mode="min", patience=20,
                                                       min_delta=0.02, restore_best_weights=True)

    # NOTE: provide the Validation Dataset so that the Model does not check itself on Train
    model.fit(wg.train, validation_data=wg.val, callbacks=[es_callback], epochs=MAX_EPOCHS)

    # Evaluate model and capture results
    eval_entry = {'PREDICTION_VAR': prediction_target.var, 'PREDICTION_LOOK_AHEAD': prediction_target.lookahead,
                  'TRAINED_TS': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'TRAIN_DATE_FROM': from_ts,
                  'TRAIN_DATE_TO': to_ts, 'TRAINING_SAMPLES': len(train_df), 'MODEL_TYPE': model_to_use, 'FEATURE_CNT': input_cnt}
    if is_binary:
        recall, precision = evaluate_classification_model(model,
                                                          wg.val,
                                                          if_all=conf.PREDICTED_VARIABLE_AGG_RULES[prediction_target.var] == 'ALL')
        eval_entry['RECALL'] = recall
        eval_entry['PRECISION'] = precision
        evaluations['CLASSIFICATION'].append(eval_entry)
        logging.info("Model for {} performance: RECALL={}, PRECISION={}".format(prediction_target, recall, precision))
    else:
        rmse, mape = evaluate_regression_model(model, wg.val)
        eval_entry['RMSE'] = rmse
        eval_entry['MAPE'] = mape
        evaluations['REGRESSION'].append(eval_entry)
        logging.info("Model for {} performance: RMSE={}, MAPE={}%".format(prediction_target, rmse, mape))

    return model, {'STD': train_std.tolist(), 'MEAN': train_mean.tolist()}


def build_model_linear(is_binary, label_width, input_cnt):
    _activation, _loss, _metrics = get_activation_loss_and_metrics(is_binary)
    model = tf.keras.Sequential([

        # Use all time steps
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([label_width, 1]),

    ])
    model.compile(loss=_loss, optimizer='adam', metrics=_metrics)

    return model


def build_model_nn(is_binary, label_width, input_cnt):
    _activation, _loss, _metrics = get_activation_loss_and_metrics(is_binary)

    model = tf.keras.Sequential([
        # Use all time steps
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=input_cnt, activation='relu'),
        tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),

        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([label_width, 1]),
    ])
    model.compile(loss=_loss, optimizer='adam', metrics=_metrics)
    return model


def build_model_dnn(is_binary, label_width, input_cnt):
    _activation, _loss, _metrics = get_activation_loss_and_metrics(is_binary)

    model = tf.keras.Sequential([
        # Use all time steps
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=input_cnt, activation='relu'),
        tf.keras.layers.Dense(units=input_cnt * 0.80, activation='relu'),

        tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),

        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([label_width, 1]),
    ])
    model.compile(loss=_loss, optimizer='adam', metrics=_metrics)
    return model


def build_model_cnn(is_binary, label_width, input_cnt):
    _activation, _loss, _metrics = get_activation_loss_and_metrics(is_binary)
    conv_width = 4

    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),

        tf.keras.layers.Conv1D(filters=input_cnt / 2,
                               kernel_size=conv_width,
                               activation='relu'),
        tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([label_width, 1]),
    ])

    model.compile(loss=_loss, optimizer='adam', metrics=[_metrics])
    return model


REGRESSION_METRICS = [tf.keras.metrics.RootMeanSquaredError()]
CLASSIFICATION_METRICS = [tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]


def get_activation_loss_and_metrics(is_binary):
    activation, loss, metrics = "linear", 'mean_squared_error', REGRESSION_METRICS
    if is_binary:
        activation, loss, metrics = "sigmoid", tf.keras.losses.BinaryCrossentropy(), CLASSIFICATION_METRICS

    return activation, loss, metrics


def evaluate_classification_model(model, test_set, if_all=True):
    predicted_labels = (model.predict(test_set, verbose=1) > 0.5).astype("int32")
    true_labels = np.concatenate([y for x, y in test_set], axis=0)

    assert len(predicted_labels) == len(true_labels)

    # We are forecasting for a number of hours:
    # aggregate each forecast series using the "True iff 1 or more is True" rule (default) or
    # "True iff all True" (option specified)
    predicted_agg = []
    true_agg = []
    for i in range(0, len(predicted_labels)):
        predicted_i = predicted_labels[i].flatten()
        true_i = true_labels[i].flatten()

        if if_all:
            predicted_i_agg = 1 if sum(predicted_i) == len(predicted_i) else 0
            true_i_agg = 1 if sum(true_i) == len(true_i) else 0
        else:
            predicted_i_agg = 1 if sum(predicted_i) > 0 else 0
            true_i_agg = 1 if sum(true_i) > 0 else 0

        predicted_agg.append(predicted_i_agg)
        true_agg.append(true_i_agg)

    recall = truncate(recall_score(true_agg, predicted_agg), 2)
    precision = truncate(precision_score(true_agg, predicted_agg), 2)

    return  recall, precision


def evaluate_regression_model(model, test_set):
    predicted_values = model.predict(test_set)
    true_values = np.concatenate([y for x, y in test_set], axis=0)

    assert len(predicted_values) == len(true_values)

    predicted_agg = []
    true_agg = []
    for i in range(0, len(predicted_values)):
        predicted_i = predicted_values[i].flatten()
        true_i = true_values[i].flatten()

        predicted_i_agg, true_i_agg = np.mean(predicted_i), np.mean(true_i)
        predicted_agg.append(predicted_i_agg)
        true_agg.append(true_i_agg)

    rmse = truncate(math.sqrt(mean_squared_error(true_agg, predicted_agg)), 2)
    mape = truncate(calc_mape(true_agg, predicted_agg), 2)

    return rmse, mape


# https://www.statology.org/mape-python/
def calc_mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    actual[abs(actual) < 0.1] = 0.1  # A meh hack to avoid division by 0

    return np.mean(np.abs((actual - pred) / actual)) * 100


# https://kodify.net/python/math/truncate-decimals/
def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


def train_models(preprocessed_readings_dir):
    conf = config.Config
    for prediction_target in conf.ALL_PREDICTION_TARGETS:
        logging.info("BEGIN: train Model for  var = {}, lookahead = {}hr".format(
            prediction_target.var, prediction_target.lookahead))
        model, norm_dict = create_model(prediction_target,
                                        conf.PREDICTION_TARGET_MODEL_TYPES[prediction_target],
                                        preprocessed_readings_dir)
        logging.info("END: train Model for  var = {}, lookahead = {}hr".format(
            prediction_target.var, prediction_target.lookahead))

        model.save(libcommons.libcommons.get_model_file(prediction_target))
        with open(libcommons.libcommons.get_normalization_file(prediction_target), "w") as outfile:
            json.dump(norm_dict, outfile)

    with open(libcommons.libcommons.get_model_info_file(), "w") as outfile:
        json.dump(evaluations, outfile)

def main(argv):
    print(argv)
    train_models(argv[1])


if __name__ == "__main__":
    conf = config.Config
    logging.basicConfig(filename=conf.TRAINER_LOG_FILE,
                        format=conf.TRAINER_LOG_FORMAT,
                        level=conf.TRAINER_LOG_LEVEL)

    main(sys.argv[0:])
