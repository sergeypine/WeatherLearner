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


def test_datastore_predictions():
    """Verify correct concatenation of new prediction and overriding with latest value"""
    ds = libcommons.DataStore()
    ds.predictions_append('2015-05-01 10:00:00', config.PredictionTarget('var1', 6), 100)
    df = ds.predictions_load()

    assert len(df) == 1
    assert df['DATE'].astype(str)[0] == '2015-05-01 10:00:00'
    assert df['LOOK_AHEAD'][0] == 6
    assert df['VAR'][0] == 'var1'
    assert df['PREDICTION'][0] == 100

    #  --- Add a row
    ds.predictions_append('2015-05-01 10:00:00', config.PredictionTarget('var1', 12), 105)
    df = ds.predictions_load()

    assert len(df) == 2
    assert df['DATE'].astype(str)[0] == '2015-05-01 10:00:00'
    assert df['LOOK_AHEAD'][0] == 6
    assert df['VAR'][0] == 'var1'
    assert df['PREDICTION'][0] == 100
    assert df['DATE'].astype(str)[1] == '2015-05-01 10:00:00'
    assert df['LOOK_AHEAD'][1] == 12
    assert df['VAR'][1] == 'var1'
    assert df['PREDICTION'][1] == 105

    # --- Override one row
    ds.predictions_append('2015-05-01 10:00:00', config.PredictionTarget('var1', 6), 95)
    df = ds.predictions_load()

    assert len(df) == 2
    assert df['DATE'].astype(str)[0] == '2015-05-01 10:00:00'
    assert df['LOOK_AHEAD'][0] == 6
    assert df['VAR'][0] == 'var1'
    assert df['PREDICTION'][0] == 95
    assert df['DATE'].astype(str)[1] == '2015-05-01 10:00:00'
    assert df['LOOK_AHEAD'][1] == 12
    assert df['VAR'][1] == 'var1'
    assert df['PREDICTION'][1] == 105


def test_feature_set_builder_build_latest():
    """Verify the building of a Pandas DataFrame containing all features needed for a model to predict
        (provide no base_time for most recent prediction) """
    ds = libcommons.DataStore()
    fsb = libcommons.FeatureSetBuilder()

    # --- Set up fake data and fake config
    dates = ['2020-02-15 09:00:00', '2020-02-15 10:00:00', '2020-02-15 11:00:00', '2020-02-15 12:00:00']
    target_df = pd.DataFrame.from_dict({'DATE': dates,
                                        'target': [0, 1, 2, 3],
                                        'incl': [0, 10, 20, 30],
                                        'excl': [0, 100, 200, 300]})
    loc1_df = pd.DataFrame.from_dict({'DATE': dates,
                                      'target': [0, 4, 5, 6],
                                      'incl': [0, 11, 21, 31],
                                      'excl': [0, 101, 201, 301]})

    loc2_df = pd.DataFrame.from_dict({'DATE': dates,
                                      'target': [0, 7, 8, 9],
                                      'incl': [0, 12, 22, 32],
                                      'excl': [0, 102, 202, 302]})
    ds.readings_append('Target', target_df)
    ds.readings_append('Loc1', loc1_df)
    ds.readings_append('Loc2', loc2_df)

    prediction_target = config.PredictionTarget('target', 2)
    config.Config.TARGET_LOCATION = 'Target'
    config.Config.PREDICTION_TARGET_FEATURES[prediction_target] = ['incl']
    config.Config.PREDICTION_TARGET_LOCATIONS[prediction_target] = ['Loc2']
    config.Config.PREDICTION_TARGET_LOOKBACKS[prediction_target] = 2

    # ---
    feature_set = fsb.build_feature_set(prediction_target)

    assert len(feature_set) == 3
    assert len(feature_set.columns) == 5
    assert 'DATE' in feature_set.columns
    assert 'target' in feature_set.columns
    assert 'target1' in feature_set.columns
    assert 'incl' in feature_set.columns
    assert 'incl1' in feature_set.columns
    assert feature_set['DATE'].astype(str)[0] == '2020-02-15 10:00:00'
    assert feature_set['DATE'].astype(str)[1] == '2020-02-15 11:00:00'
    assert feature_set['DATE'].astype(str)[2] == '2020-02-15 12:00:00'
    assert feature_set['target'][0] == 1
    assert feature_set['target'][1] == 2
    assert feature_set['target'][2] == 3
    assert feature_set['target1'][0] == 7
    assert feature_set['target1'][1] == 8
    assert feature_set['target1'][2] == 9
    assert feature_set['incl'][0] == 10
    assert feature_set['incl'][1] == 20
    assert feature_set['incl'][2] == 30
    assert feature_set['incl1'][0] == 12
    assert feature_set['incl1'][1] == 22
    assert feature_set['incl1'][2] == 32


def test_feature_set_builder_build_base_time():
    """Verify the building of a Pandas DataFrame containing all features needed for a model to predict
        (provide  base_time for non- recent prediction) """
    ds = libcommons.DataStore()
    fsb = libcommons.FeatureSetBuilder()

    # --- Set up fake data and fake config
    dates = ['2020-02-15 10:00:00', '2020-02-15 11:00:00', '2020-02-15 12:00:00', '2020-02-15 13:00:00']
    target_df = pd.DataFrame.from_dict({'DATE': dates,
                                        'target': [1, 2, 3, 4],
                                        'incl': [10, 20, 30, 40],
                                        'excl': [100, 200, 300, 400]})
    loc1_df = pd.DataFrame.from_dict({'DATE': dates,
                                      'target': [5, 6, 7, 8],
                                      'incl': [11, 21, 31, 41],
                                      'excl': [101, 201, 301, 401]})

    ds.readings_append('Target_2', target_df)
    ds.readings_append('Loc1_2', loc1_df)

    prediction_target = config.PredictionTarget('target', 1)
    config.Config.TARGET_LOCATION = 'Target_2'
    config.Config.PREDICTION_TARGET_FEATURES[prediction_target] = ['incl']
    config.Config.PREDICTION_TARGET_LOCATIONS[prediction_target] = ['Loc1_2']
    config.Config.PREDICTION_TARGET_LOOKBACKS[prediction_target] = 1

    # ---
    feature_set = fsb.build_feature_set(prediction_target, pd.Timestamp('2020-02-15 12:00:00'))

    assert len(feature_set) == 2
    assert len(feature_set.columns) == 5
    assert 'DATE' in feature_set.columns
    assert 'target' in feature_set.columns
    assert 'target1' in feature_set.columns
    assert 'incl' in feature_set.columns
    assert 'incl1' in feature_set.columns
    assert feature_set['DATE'].astype(str)[0] == '2020-02-15 11:00:00'
    assert feature_set['DATE'].astype(str)[1] == '2020-02-15 12:00:00'
    assert feature_set['target'][0] == 2
    assert feature_set['target'][1] == 3
    assert feature_set['target1'][0] == 6
    assert feature_set['target1'][1] == 7
    assert feature_set['incl'][0] == 20
    assert feature_set['incl'][1] == 30
    assert feature_set['incl1'][0] == 21
    assert feature_set['incl1'][1] == 31


def test_feature_set_builder_build_missing_tail():
    """Verify the building of a Pandas DataFrame still works if adjacent locations miss most recent reading """
    ds = libcommons.DataStore()
    fsb = libcommons.FeatureSetBuilder()

    # --- Set up fake data and fake config
    target_df = pd.DataFrame.from_dict({'DATE': ['2020-02-15 10:00:00', '2020-02-15 11:00:00', '2020-02-15 12:00:00'],
                                        'target': [1, 2, 3],
                                        'incl': [10, 20, 30],
                                        'excl': [100, 200, 300]})
    loc1_df = pd.DataFrame.from_dict({'DATE': ['2020-02-15 10:00:00', '2020-02-15 11:00:00'],
                                      'target': [5, 6],
                                      'incl': [11, 21],
                                      'excl': [101, 201]})

    ds.readings_append('Target_3', target_df)
    ds.readings_append('Loc1_3', loc1_df)

    prediction_target = config.PredictionTarget('target', 1)
    config.Config.TARGET_LOCATION = 'Target_3'
    config.Config.PREDICTION_TARGET_FEATURES[prediction_target] = ['incl']
    config.Config.PREDICTION_TARGET_LOCATIONS[prediction_target] = ['Loc1_3']
    config.Config.PREDICTION_TARGET_LOOKBACKS[prediction_target] = 2

    # ---
    feature_set = fsb.build_feature_set(prediction_target)
    print(feature_set)
    assert len(feature_set) == 3
    assert len(feature_set.columns) == 5
    assert 'DATE' in feature_set.columns
    assert 'target' in feature_set.columns
    assert 'target1' in feature_set.columns
    assert 'incl' in feature_set.columns
    assert 'incl1' in feature_set.columns
    assert feature_set['DATE'].astype(str)[0] == '2020-02-15 10:00:00'
    assert feature_set['DATE'].astype(str)[1] == '2020-02-15 11:00:00'
    assert feature_set['DATE'].astype(str)[2] == '2020-02-15 12:00:00'
    assert feature_set['target'][0] == 1
    assert feature_set['target'][1] == 2
    assert feature_set['target'][2] == 3
    assert feature_set['target1'][0] == 5
    assert feature_set['target1'][1] == 6
    assert feature_set['target1'][2] == 6
    assert feature_set['incl'][0] == 10
    assert feature_set['incl'][1] == 20
    assert feature_set['incl'][2] == 30
    assert feature_set['incl1'][0] == 11
    assert feature_set['incl1'][1] == 21
    assert feature_set['incl1'][2] == 21

