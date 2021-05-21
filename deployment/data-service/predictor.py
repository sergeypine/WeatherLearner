import sys
import datetime
import logging
import json
import itertools
import pandas as pd
import numpy as np
from tensorflow import keras

sys.path.insert(1, '../')
import config
import libcommons.libcommons


class Predictor():
    def __init__(self):
        self.conf = config.Config
        self.data_store = libcommons.libcommons.DataStore()
        self.feature_set_builder = libcommons.libcommons.FeatureSetBuilder()

        # pre-load all the models
        self.models = {}
        self.means_and_stds = {}
        for prediction_target in self.conf.ALL_PREDICTION_TARGETS:
            # Model
            model = keras.models.load_model(libcommons.libcommons.get_model_file(prediction_target))
            logging.info("Loaded Model for var = {}, lookahead = {}hr".format(prediction_target.var,
                                                                              prediction_target.lookahead))
            logging.info(model.summary())
            self.models[prediction_target] = model

            # Mean & Std
            with open(libcommons.libcommons.get_normalization_file(prediction_target)) as json_file:
                mean_std_dict = json.load(json_file)
                self.means_and_stds[prediction_target] = [mean_std_dict['MEAN'], mean_std_dict['STD']]

    def predict_for_target_and_base_time(self, prediction_target: config.PredictionTarget,
                                         base_time: datetime.datetime = None):
        # Get a Dataframe with data for locations and feature vars that are relevant to this prediction target
        feature_set = self.feature_set_builder.build_feature_set(prediction_target)

        # TODO - manage base_time!

        # Normalize Data
        feature_set = self.feature_set_builder.normalize_data(feature_set, prediction_target,
                                                              *self.means_and_stds[prediction_target])

        # Get the right model for this prediction
        model = self.models[prediction_target]

        # Generate a tensor that can be fed into that model to make prediction
        model_tensor = self.make_tensor(feature_set, prediction_target)

        # Perform the prediction and record the result
        prediction = self.predict(prediction_target, model_tensor, model)
        prediction_time = base_time + datetime.timedelta(hours=prediction_target.lookahead)
        logging.info("Target {}, prediction_time = {}; predicted value {}".format(
            prediction_target, prediction_time, prediction))

        # Add this prediction to Data Store
        self.data_store.predictions_append(prediction_time, prediction_target, prediction)

    def make_tensor(self, featureset, prediction_target):
        # Remove all rows except for the ones needed to make prediction (last lookback hours rows)
        featureset = featureset.tail(self.conf.PREDICTION_TARGET_LOOKBACKS[prediction_target])

        # Add dummy rows up to lookahead time (WindowGenerator class needs to have rows for the forecast timeframe)
        for _ in itertools.repeat(None, prediction_target.lookahead -
                                        self.conf.PREDICTED_VARIABLE_AHI[prediction_target.var]):
            featureset = featureset.append(pd.Series(), ignore_index=True)

        agg_interval = 2 * self.conf.PREDICTED_VARIABLE_AHI[prediction_target.var] + 1
        wg = libcommons.libcommons.WindowGenerator(
            input_width=self.conf.PREDICTION_TARGET_LOOKBACKS[prediction_target],
            label_width=agg_interval,
            shift=prediction_target.lookahead -
                  self.conf.PREDICTED_VARIABLE_AHI[prediction_target.var][prediction_target.var],
            label_columns=[prediction_target.var],
            train_df=featureset,
            test_df=None,
            val_df=None)

        t = wg.train.__iter__().get_next()[0]
        return t

    def predict(self, prediction_target, model_tensor, model):
        agg_rule = self.conf.PREDICTED_VARIABLE_AGG_RULES[prediction_target.var]
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