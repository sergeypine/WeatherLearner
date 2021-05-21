import sys
import os
import datetime
import shutil

import pandas as pd
import pytest

sys.path.insert(1, '../')

import config
import libcommons


@pytest.fixture(scope="session", autouse=True)
def common(request):
    def remove_test_dir():
        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

    remove_test_dir()
    request.addfinalizer(remove_test_dir)


def test_datastore_readings():
    """Verify correct concatenation of new readings and that duplicates are eliminated"""
    ds = libcommons.DataStore()
    ds.readings_append('Rio', pd.DataFrame.from_dict({'DATE': ['2020-02-15 10:00:00'], 'data': ['data1']}))
    df = ds.readings_load('Rio')
    assert len(df) == 1

    ds.readings_append('Rio', pd.DataFrame.from_dict(
        {'DATE': ['2020-02-15 10:00:00', '2020-02-15 11:00:00'], 'data': ['data1', 'data2']}))
    df = ds.readings_load('Rio')
    assert len(df) == 2
    assert df.iloc[0]['data'] == 'data1'
    assert df.iloc[1]['data'] == 'data2'


def test_feature_set_builder_build():
    """Verify the building of a Pandas DataFrame containing all features needed for a model to predict"""
    ds = libcommons.DataStore()
    fsb = libcommons.FeatureSetBuilder()

    # --- Set up fake data and fake config
    target_df = pd.DataFrame.from_dict({'DATE': ['2020-02-15 10:00:00', '2020-02-15 11:00:00', '2020-02-15 12:00:00'],
                                        'target': [1, 2, 3],
                                        'incl': [10, 20, 30],
                                        'excl': [100, 200, 300]})
    loc1_df = pd.DataFrame.from_dict({'DATE': ['2020-02-15 10:00:00', '2020-02-15 11:00:00', '2020-02-15 12:00:00'],
                                      'target': [4, 5, 6],
                                      'incl': [11, 21, 31],
                                      'excl': [101, 201, 301]})

    loc2_df = pd.DataFrame.from_dict({'DATE': ['2020-02-15 10:00:00', '2020-02-15 11:00:00'],
                                      'target': [7, 8],
                                      'incl': [12, 22],
                                      'excl': [102, 202]})
    ds.readings_append('Target', target_df)
    ds.readings_append('Loc1', loc1_df)
    ds.readings_append('Loc2', loc2_df)

    prediction_target = config.PredictionTarget('target', 2)
    config.Config.TARGET_LOCATION = 'Target'
    config.Config.PREDICTION_TARGET_FEATURES[prediction_target] = ['incl']
    config.Config.PREDICTION_TARGET_LOCATIONS[prediction_target] = ['Loc2']

    # ---
    feature_set = fsb.build_feature_set(prediction_target)

    assert len(feature_set.columns) == 4
    assert 'target' in feature_set.columns
    assert 'target1' in feature_set.columns
    assert 'incl' in feature_set.columns
    assert 'incl1' in feature_set.columns
    assert feature_set['target'][0] == 1
    assert feature_set['target'][1] == 2
    assert feature_set['target1'][0] == 7
    assert feature_set['target1'][1] == 8
    assert feature_set['incl'][0] == 10
    assert feature_set['incl'][1] == 20
    assert feature_set['incl1'][0] == 12
    assert feature_set['incl1'][1] == 22
