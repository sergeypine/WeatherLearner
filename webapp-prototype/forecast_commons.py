import tensorflow as tf
import numpy as np
import pandas as pd
from flask import current_app


def get_normalization_file(prediction_target):
    return "pretrained/{}_{}h_normalization.json".format(prediction_target.var, prediction_target.lookahead)


def get_model_file(prediction_target):
    return "pretrained/{}_{}h.h5".format(prediction_target.var, prediction_target.lookahead)


def get_latest_predictions_file():
    return "predictions/latest.csv"


# Mostly copy-paste of https://www.tensorflow.org/tutorials/structured_data/time_series
class WindowGenerator():

    def __init__(self,
                 input_width,  # Lookback Window (hours into the past to base predictions on)
                 label_width,  # Aggregation Interval (how many hours of data we'll be predicting)
                 shift,  # How many hours in advance we'll be predicting
                 train_df, val_df, test_df,  # Training, Validation and Testing sets
                 label_columns=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


# =====================================================================
# ======================================================================
def build_feature_set(prediction_target, location_file_format_str):
    target_df = load_location_readings(current_app.config['TARGET_LOCATION'], location_file_format_str)
    target_df = drop_unused_columns(target_df, prediction_target)
    merged_df = target_df
    suffix_no = 1

    # Merge adjacent location files one by one relying on DATA
    for adjacent_location in current_app.config['PREDICTION_TARGET_LOCATIONS'][prediction_target]:
        adjacent_df = load_location_readings(adjacent_location, location_file_format_str)
        adjacent_df = drop_unused_columns(adjacent_df, prediction_target)

        # Take control of column name suffix in the dataset being merged in
        adjacent_df = adjacent_df.add_suffix(str(suffix_no))
        adjacent_df = adjacent_df.rename(columns={"DATE{}".format(suffix_no): 'DATE'})
        merged_df = pd.merge(merged_df, adjacent_df, on='DATE')
        suffix_no = suffix_no + 1

    # DATE column is of no use in the modelling stage
    merged_df = merged_df.drop(columns=['DATE'])

    return merged_df


def load_location_readings(location, location_file_format_str):
    loc_readings_df = pd.read_csv(location_file_format_str.format(location))
    return loc_readings_df


def drop_unused_columns(df, prediction_target):
    features_to_use = current_app.config['PREDICTION_TARGET_FEATURES'][prediction_target]
    all_columns = features_to_use.copy()
    all_columns.append('DATE')
    all_columns.append(prediction_target.var)
    df = df[all_columns]

    return df


def normalize_data(featureset, prediction_target, mean, std):
    columns_to_normalize = get_columns_to_normalize(featureset, prediction_target)
    featureset[columns_to_normalize] = (featureset[columns_to_normalize] - mean) / std
    return featureset


def get_columns_to_normalize(featureset, prediction_target):
    return featureset.columns.drop([prediction_target.var])
