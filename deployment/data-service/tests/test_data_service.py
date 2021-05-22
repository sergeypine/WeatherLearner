import sys
import os
import datetime
import shutil
import pytest

sys.path.insert(1, '../')

import reading_retriever
import predictor
import config
import libcommons.libcommons


@pytest.fixture(scope="session", autouse=True)
def common(request):
    def remove_test_dir():
        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

    remove_test_dir()
    request.addfinalizer(remove_test_dir)


# @pytest.mark.skip(reason="save time")
def test_reading_retriever():
    """ Retrieve weather data for 2 days from wundeground, then sanity check data"""
    rr = reading_retriever.ReadingRetriever()
    ds = libcommons.libcommons.DataStore()

    rr.retrieve_for_date_and_location(datetime.datetime.strptime('2015-06-15', '%Y-%m-%d'), 'Chicago')
    df = ds.readings_load('Chicago')
    assert len(df) == 24  # Exactly 24 entries

    rr.retrieve_for_date_and_location(datetime.datetime.strptime('2015-06-16', '%Y-%m-%d'), 'Chicago')
    df = ds.readings_load('Chicago')
    assert len(df) == 48  # New date entries must be appended

    # Check date ranges are correct
    df['DATE'] = df['DATE'].astype(str)  # blegh
    assert df['DATE'][0] == '2015-06-15 00:00:00'
    assert df['DATE'][23] == '2015-06-15 23:00:00'
    assert df['DATE'].iloc[24] == '2015-06-16 00:00:00'
    assert df['DATE'].iloc[-1] == '2015-06-16 23:00:00'

    # Spot check one hourly reading
    target_row = df.iloc[45]
    assert target_row['DATE'] == '2015-06-16 21:00:00'
    assert target_row['Temp'] == 61
    assert target_row['DewPoint'] == 46
    assert target_row['Humidity'] == 58
    assert target_row['WindSpeed'] == 9
    assert target_row['Pressure'] == 29.47
    assert target_row['_cloud_intensity'] == 4.0
    assert target_row['_is_clear'] == 0.0
    assert target_row['_is_precip'] == 0.0
    print(df)


def test_predictor():
    """Verify that predictor runs for all prediction targets and produces non-insane predictions
        (NOTE: this also verifies all pretrained models"""
    old_ds_path = config.Config.DATA_STORE_BASE_DIR
    config.Config.DATA_STORE_BASE_DIR = "tests/test_data"
    if os.path.exists("{}/predictions.csv".format(config.Config.DATA_STORE_BASE_DIR)):
        os.remove("{}/predictions.csv".format(config.Config.DATA_STORE_BASE_DIR))

    pr = predictor.Predictor()
    ds = libcommons.libcommons.DataStore()

    try:
        for prediction_target in config.Config.ALL_PREDICTION_TARGETS:
            pr.predict_for_target_and_base_time(prediction_target)
        predictions = ds.predictions_load()

        assert len(predictions) == len(config.Config.ALL_PREDICTION_TARGETS)

        for prediction_target in config.Config.ALL_PREDICTION_TARGETS:
            prediction_row = predictions[(predictions['VAR'] == prediction_target.var) &
                                         (predictions['LOOK_AHEAD'] == prediction_target.lookahead)].iloc[0]
            prediction = prediction_row['PREDICTION']
            if prediction_target.var == 'Temp':
                assert 50 < prediction < 100
            elif prediction_target.var == 'WindSpeed':
                assert 0 < prediction < 20
            elif prediction_target.var in ['_is_clear', '_is_precip']:
                assert prediction in [0, 1]

    finally:
        config.Config.DATA_STORE_BASE_DIR = old_ds_path
