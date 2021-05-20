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
    rr = reading_retriever.ReadingRetriever()
    ds = libcommons.libcommons.DataStore()

    rr.retrieve_for_date_and_location(datetime.datetime.strptime('2020-06-15', '%Y-%m-%d'), 'Chicago')
    df = ds.readings_load('Chicago')
    assert len(df) == 24

    rr.retrieve_for_date_and_location(datetime.datetime.strptime('2020-06-16', '%Y-%m-%d'), 'Chicago')
    df = ds.readings_load('Chicago')
    assert len(df) == 48
    assert df['DATE'][0] == '2020-06-15 01:00:00'
    assert df['DATE'].iloc[-1] == '2020-06-16 23:00:00'