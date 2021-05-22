import datetime
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


# =============================================================================
# =============================================================================


class DataStore(object):

    def __init__(self):
        self.conf = config.Config

    def readings_append(self, location, data_frame: pd.DataFrame):
        target_file = "{}/readings/{}.csv".format(self.conf.DATA_STORE_BASE_DIR, location)
        if not path.exists(target_file):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            data_frame.to_csv(target_file, index=False)
        else:
            existing_df = pd.read_csv(target_file, parse_dates=['DATE'])

            data_frame['DATE'] = pd.to_datetime(data_frame['DATE'])
            existing_df['DATE'] = pd.to_datetime(existing_df['DATE'])

            # Chuck all readings for timestamps present in this new frame
            dates_in_appended = np.array(data_frame['DATE'].unique().astype('datetime64[ns]'))
            existing_df = existing_df[~(existing_df['DATE'].isin(dates_in_appended))]

            # Now add in the new frame
            existing_df = pd.concat([existing_df, data_frame], ignore_index=True)

            existing_df = existing_df.drop_duplicates(subset=['DATE'])
            existing_df = existing_df.sort_values(by='DATE')

            existing_df.to_csv(target_file, index=False)

    def readings_load(self, location):
        target_file = "{}/readings/{}.csv".format(self.conf.DATA_STORE_BASE_DIR, location)
        return pd.read_csv(target_file, parse_dates=['DATE'])

    def predictions_append(self, prediction_time, prediction_target, predicted_val):
        target_file = "{}/predictions.csv".format(self.conf.DATA_STORE_BASE_DIR)
        df = pd.DataFrame.from_dict({
            'DATE': [prediction_time],
            'LOOK_AHEAD': [prediction_target.lookahead],
            'VAR': [prediction_target.var],
            'PREDICTION': [predicted_val]
        })
        if not path.exists(target_file):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            df.to_csv(target_file, index=False)
        else:
            existing_df = pd.read_csv(target_file, parse_dates=['DATE'])

            # Override the collision, if any, with this latest prediction
            existing_df = existing_df.drop(
                existing_df[(existing_df['DATE'] == prediction_time) &
                            (existing_df['VAR'] == prediction_target.var) &
                            (existing_df['LOOK_AHEAD'] == prediction_target.lookahead)].index)

            existing_df = pd.concat([existing_df, df], ignore_index=True)
            existing_df = existing_df.drop_duplicates(subset=['DATE', 'VAR', 'PREDICTION'])

            existing_df['DATE'] = pd.to_datetime(existing_df['DATE'])
            existing_df = existing_df.sort_values(by=['DATE', 'LOOK_AHEAD'])

            existing_df.to_csv(target_file, index=False)

    def predictions_load(self):
        target_file = "{}/predictions.csv".format(self.conf.DATA_STORE_BASE_DIR)
        return pd.read_csv(target_file, parse_dates=['DATE'])
        pass


# =============================================================================
# =============================================================================

class FeatureSetBuilder(object):
    def __init__(self):
        self.conf = config.Config
        self.data_store = DataStore()

    def build_feature_set(self, prediction_target, base_time=None):
        """
        Build a Pandas DF containing all features for a prediction target, one row/timestamp

        :param prediction_target: what we are predicting (predicted variable, lookahead time)
        :param base_time: timestamp of the latest reading based on which we are predicting
            (if None, use latest available)
        :return: Pandas DF containing all features for a prediction target, one row/timestamp, no extra rows
        """
        target_df = self.data_store.readings_load(self.conf.TARGET_LOCATION)
        target_df = self.drop_unused_columns(target_df, prediction_target)

        # Base Time not provided -> assume we are predicting for the latest reading
        if base_time is None:
            base_time = target_df.iloc[-1]['DATE']

        #  get rid of rows outside time range & pad if timestamps are missing
        merged_df = self.adjust_df_to_relevant_time_range(prediction_target, target_df, base_time)

        suffix_no = 1

        # Merge adjacent location files one by one relying on DATE
        for adjacent_location in self.conf.PREDICTION_TARGET_LOCATIONS[prediction_target]:
            adjacent_df = self.data_store.readings_load(adjacent_location)

            #  get rid of rows outside time range & pad if timestamps are missing
            adjacent_df = self.adjust_df_to_relevant_time_range(prediction_target, adjacent_df, base_time)

            adjacent_df = self.drop_unused_columns(adjacent_df, prediction_target)

            # Take control of column name suffix in the dataset being merged in
            adjacent_df = adjacent_df.add_suffix(str(suffix_no))
            adjacent_df = adjacent_df.rename(columns={"DATE{}".format(suffix_no): 'DATE'})
            merged_df = pd.merge(merged_df, adjacent_df, on='DATE')
            suffix_no = suffix_no + 1

        return merged_df

    def drop_unused_columns(self, df, prediction_target):
        features_to_use = self.conf.PREDICTION_TARGET_FEATURES[prediction_target]
        all_columns = features_to_use.copy()
        all_columns.append('DATE')
        all_columns.append(prediction_target.var)
        df = df[all_columns]

        return df

    def adjust_df_to_relevant_time_range(self, prediction_target, df, base_time):
        df = df[df['DATE'] <= base_time]
        df = df.tail(self.conf.PREDICTION_TARGET_LOOKBACKS[prediction_target] + 1)

        # If last row is not @ base_time, pad the tail of DF with duplicates of last row
        missing_hrs = int((base_time - df.iloc[-1]['DATE']).seconds / 3600)

        while missing_hrs > 0:
            df = df.append(pd.Series(df.iloc[-1]), ignore_index=True)
            df.loc[df.index[-1], 'DATE'] = df.iloc[-1]['DATE'] + datetime.timedelta(hours=1)
            missing_hrs = missing_hrs - 1

        return df

    def normalize_data(self, featureset, prediction_target, mean, std):
        columns_to_normalize = self.get_columns_to_normalize(featureset, prediction_target)
        featureset[columns_to_normalize] = (featureset[columns_to_normalize] - mean) / std
        return featureset

    @staticmethod
    def get_columns_to_normalize(featureset, prediction_target):
        return featureset.columns.drop([prediction_target.var])


# =============================================================================
# =============================================================================
def get_normalization_file(prediction_target):
    return "{}/{}_{}h_normalization.json".format(config.Config.MODELS_BASE_DIR,
                                                 prediction_target.var, prediction_target.lookahead)


def get_model_file(prediction_target):
    return "{}/{}_{}h.h5".format(config.Config.MODELS_BASE_DIR, prediction_target.var, prediction_target.lookahead)