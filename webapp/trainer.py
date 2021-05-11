import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow import keras
import json
import forecast_commons
from flask import current_app


def create_model(prediction_target, model_to_use):
    # Load the Target Location and Adjacent Locations Datasets and drop the unused columns
    featureset = forecast_commons.build_feature_set(prediction_target,
                                                    current_app.config['LOCATION_DATASET_FILE_FORMAT'])

    # Split the data: 70% for training, 30% for validation
    n = len(featureset)
    train_df = featureset[0: int(n * 0.70)]
    val_df = featureset[int(n * 0.70):]

    # Normalize input data (NOTE: TF tutorial also scales the target variable)
    train_mean = train_df[forecast_commons.get_columns_to_normalize(train_df, prediction_target)].mean()
    train_std = train_df[forecast_commons.get_columns_to_normalize(train_df, prediction_target)].std()
    train_df = forecast_commons.normalize_data(train_df, prediction_target, train_mean, train_std)
    val_df = forecast_commons.normalize_data(val_df, prediction_target, train_mean, train_std)

    # Generate Timeseries Prediction Window on which TensorFlow will operate
    ahi_interval = current_app.config['PREDICTED_VARIABLE_AHI'][prediction_target.var]
    agg_interval = 2 * ahi_interval + 1
    wg = forecast_commons.WindowGenerator(
        input_width=current_app.config['PREDICTION_TARGET_LOOKBACKS'][prediction_target],
        label_width=agg_interval,
        shift=prediction_target.lookahead - ahi_interval,
        label_columns=[prediction_target.var],
        train_df=train_df, val_df=val_df, test_df=None)

    # Build the desired model
    input_cnt = (len(current_app.config['PREDICTION_TARGET_LOCATIONS'][prediction_target]) + 1) * \
                (len(current_app.config['PREDICTION_TARGET_FEATURES'][prediction_target]) + 1) * \
                current_app.config['PREDICTION_TARGET_LOOKBACKS'][prediction_target]
    current_app.logger.info("Input Cnt = {}".format(input_cnt))
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
    model.fit(wg.train, validation_data=wg.val, callbacks=[es_callback], epochs=1)

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


def get_activation_loss_and_metrics(isBinary):
    activation, loss, metrics = "linear", 'mean_squared_error', REGRESSION_METRICS
    if isBinary:
        activation, loss, metrics = "sigmoid", tf.keras.losses.BinaryCrossentropy(), CLASSIFICATION_METRICS

    return activation, loss, metrics


def train_models():
    for prediction_target in current_app.config['ALL_PREDICTION_TARGETS']:
        current_app.logger.info("BEGIN: train Model for  var = {}, lookahead = {}hr".format(
            prediction_target.var, prediction_target.lookahead))
        model, norm_dict = create_model(prediction_target,
                                        current_app.config['PREDICTION_TARGET_MODEL_TYPES'][prediction_target])
        current_app.logger.info("END: train Model for  var = {}, lookahead = {}hr".format(
            prediction_target.var, prediction_target.lookahead))

        model.save(forecast_commons.get_model_file(prediction_target))
        with open(forecast_commons.get_normalization_file(prediction_target), "w") as outfile:
            json.dump(norm_dict, outfile)

