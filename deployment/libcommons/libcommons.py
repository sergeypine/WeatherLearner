import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import os.path
from os import path

sys.path.insert(1, '../')
import config

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


class DataStore(object):

    def __init__(self):
        self.conf = config.Config

    def readings_append(self, location, data_frame):
        target_file = "{}/readings/{}.csv".format(self.conf.DATA_STORE_BASE_DIR, location)
        if not path.exists(target_file):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            data_frame.to_csv(target_file)
        else:
            data_frame.to_csv(target_file, mode='a', header=False)

    def readings_load(self, location):
        target_file = "{}/readings/{}.csv".format(self.conf.DATA_STORE_BASE_DIR, location)
        return pd.read_csv(target_file)
