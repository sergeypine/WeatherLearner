import sys
import os
import datetime
import shutil
import pytest

sys.path.insert(1, '../')

import reading_retriever
import config
import libcommons.libcommons


@pytest.fixture(scope="session", autouse=True)
def common(request):
    def remove_test_dir():
        if os.path.exists(config.Config.DATA_STORE_BASE_DIR):
            shutil.rmtree(config.Config.DATA_STORE_BASE_DIR)

    remove_test_dir()
    request.addfinalizer(remove_test_dir)


def test_reading_retriever():
    """ Retrieve weather data for 2 days from wundeground, then sanity check it"""
    rr = reading_retriever.ReadingRetriever()
    ds = libcommons.libcommons.DataStore()

    rr.retrieve_for_date_and_location(datetime.datetime.strptime('2015-06-15', '%Y-%m-%d'), 'Chicago')
    df = ds.readings_load('Chicago')
    assert len(df) == 24  # Exactly 24 entries

    rr.retrieve_for_date_and_location(datetime.datetime.strptime('2015-06-16', '%Y-%m-%d'), 'Chicago')
    df = ds.readings_load('Chicago')
    assert len(df) == 48  # New date entries must be appended

    # Check date ranges are correct
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
