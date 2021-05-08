import numpy as np
from flask import Flask
import time
import datetime
import pytz
import random
import itertools
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import sys
import csv
from selenium.webdriver.chrome.options import Options
from flask import current_app
import logging

app = Flask(__name__)

PAGE_LOAD_RETRIES = 4
WAIT_TIME = 2
chrome_options = Options()
chrome_options.headless = True
_driver = webdriver.Chrome(executable_path="bin/chromedriver",
                           chrome_options=chrome_options)


def retrieve_latest_readings():
    location_codes = current_app.config['LOCATION_CODES']
    for location, location_code in location_codes.items():
        current_app.logger.info("Retrieving readings for {}".format(location))
        loc_data = download_loc_data(location_code)
        loc_data_noaa = loc_data_to_noo_format(loc_data)
        save_loc_data(location, loc_data_noaa)

    current_app.logger.info("Finished retrieving readings, location cnt = {}".format(len(location_codes)))
    return []


def download_loc_data(location_code):
    recent_dates = get_applicable_date_list()
    hourly_data = []
    for recent_date in recent_dates:
        html = get_html_for_date(recent_date, location_code)
        hourly_data.append(get_hourly_for_date_from_html(html, recent_date))

    column_names = ['Time', 'Temp', 'DewPoint', 'Humidity', 'Wind', 'WindSpeed', 'WindGust', 'Pressure', 'Precip',
                    'Condition']
    return pd.DataFrame(hourly_data[0], columns=column_names)


def loc_data_to_noo_format(loc_data: pd.DataFrame):
    # Drop "F" in temperatures
    loc_data['Temp'] = loc_data['Temp'].str.replace(' F', '')
    loc_data['DewPoint'] = loc_data['DewPoint'].str.replace(' F', '')

    # Drop % in Humidty
    loc_data['Humidity'] = loc_data['Humidity'].str.replace(' %', '')

    # Drop mph in Windspeed and WindGust
    loc_data['WindSpeed'] = loc_data['WindSpeed'].str.replace(' mph', '')
    loc_data['WindGust'] = loc_data['WindGust'].str.replace(' mph', '')

    # Drop in from Pressure and Precip
    loc_data['Pressure'] = loc_data['Pressure'].str.replace(' in', '')
    loc_data['Precip'] = loc_data['Precip'].str.replace(' in', '')

    # Transform wind direction into 3 components: Northerly, Easterly and is_var
    loc_data['_wind_dir_sin'], loc_data['_wind_dir_cos'] = zip(
        *loc_data['Wind'].apply(get_wind_dir_sin_cos))

    # Transform condition into numeric components
    loc_data['_is_clear'], loc_data['_is_precip'], loc_data['_cloud_intensity'], loc_data['_is_thunder'], \
        loc_data['_is_snow'], loc_data['_is_mist'] = zip(*loc_data['Condition'].apply(get_condition_components))

    # Best guess for the following data missing in wundeground but present in NOAA
    # TODO - this seems sloppy, think of a better solution
    loc_data['CloudAltitude'] = loc_data['_is_clear'].apply(lambda x: 30000 if x == 1 else 7000)
    loc_data['Visibility'] = loc_data['_is_mist'].apply(lambda x: 5 if x == 1 else 10)
    loc_data['PressureChange'] = 0  # blegh

    # ************
    cleaned_df = loc_data
    # *************

    # Round all timestamps to 1h
    cleaned_df['DATE'] = pd.to_datetime(cleaned_df['Time']).dt.round('1h')

    cleaned_df = cleaned_df.sort_values(by='DATE')

    # Drop rows with identical timestamp (for some hours we have multiple readings, keep one)
    # NOTE - this chooses a random duplicate value, consider doing an average instead
    cleaned_df = cleaned_df.drop_duplicates(subset=['DATE'])

    # Fill in missing hours and impune values but only within 3h
    cleaned_df = cleaned_df.set_index('DATE').resample('H').nearest(limit=3).dropna().reset_index()

    # Create day-of-year and hour-of-day sin/cos columns to capture the cyclical nature of time
    cleaned_df['_day_sin'] = np.sin(2 * np.pi * cleaned_df['DATE'].dt.dayofyear / 365)
    cleaned_df['_day_cos'] = np.cos(2 * np.pi * cleaned_df['DATE'].dt.dayofyear / 365)

    cleaned_df['_hour_sin'] = np.sin(2 * np.pi * cleaned_df['DATE'].dt.hour / 24)
    cleaned_df['_hour_cos'] = np.cos(2 * np.pi * cleaned_df['DATE'].dt.hour / 24)

    # Shed the unnecessary columns
    cleaned_df = cleaned_df.drop(columns=['Time', 'Condition', 'Wind'])
    return cleaned_df


def save_loc_data(location, loc_data):
    loc_data.to_csv("readings/{}.csv".format(location))


def get_html_for_date(date, location):
    # URL example: https://www.wunderground.com/history/daily/KMDW/date/2015-1-21
    target_url = 'https://www.wunderground.com/history/daily/{}/date/{}'.format(location, date)

    attempts = 0

    while attempts < PAGE_LOAD_RETRIES:
        try:
            _driver.get(target_url)
            time.sleep(WAIT_TIME)  # blegh...

            soup = BeautifulSoup(_driver.page_source, 'lxml')
            table_html = soup.find_all('table')[1]
            return table_html
        except:
            attempts = attempts + 1
            if attempts >= PAGE_LOAD_RETRIES:
                print("Unexpected error loading page:", sys.exc_info()[0], sys.exc_info()[1])

    return None


def get_hourly_for_date_from_html(page_html, date):
    hourly_data_list = []

    df = pd.read_html(str(page_html))[0]

    # get rid of NAN's
    df = df.dropna()

    # prepend date  to time
    df['Time'] = df['Time'].apply(lambda x: "{} {}".format(date, x))

    # Iterate over each row
    for index, row in df.iterrows():
        hourly_data_list.append([
            row['Time'],
            row['Temperature'].replace("\xa0", " "),
            row['Dew Point'].replace("\xa0", " "),
            row['Humidity'].replace("\xa0", " "),
            row['Wind'].replace("\xa0", " "),
            row['Wind Speed'].replace("\xa0", " "),
            row['Wind Gust'].replace("\xa0", " "),
            row['Pressure'].replace("\xa0", " "),
            row['Precip.'].replace("\xa0", " "),
            row['Condition'].replace("\xa0", " ")])

    return hourly_data_list


def get_applicable_date_list():
    now_datetime = datetime.datetime.now(pytz.timezone(current_app.config['TARGET_TIMEZONE']))
    lookback_datetime = now_datetime - datetime.timedelta(hours=current_app.config['MAX_LOOK_BACK_HOURS'])

    applicable_date_list = [lookback_datetime.date()]
    if (now_datetime - lookback_datetime).days > 0:
        applicable_date_list.append(now_datetime.date())
    # NOTE - won't work for more than 24h lookbacks

    return applicable_date_list


def get_wind_dir_sin_cos(wind_direction):
    wind_dir_angle_dict = {
        'W': 348.75,
        'WSW': 236.25,
        'SSW': 191.25,
        'SW': 213.75,
        'S': 168.75,
        'WNW': 281.25,
        'NW': 303.75,
        'CALM': 0,
        'NNW': 326.25,
        'N': 348.75,
        'VAR': 0,
        'ENE': 56.25,
        'NNE': 11.25,
        'ESE': 101.25,
        'SSE': 146.25,
        'SE': 123.75,
        'E': 78.75,
        'NE': 33.75
    }
    return np.sin(2 * np.pi * wind_dir_angle_dict[wind_direction] / 360), \
           np.cos(2 * np.pi * wind_dir_angle_dict[wind_direction] / 360)


def get_condition_components(condition):
    is_clear = 0
    is_precip = 0
    cloud_intensity = 0
    is_thunder = 0
    is_snow = 0
    is_mist = 0

    lcondition = condition.lower()

    if ('fair' in lcondition) or ('smoke' in lcondition) or ('haze' in lcondition):
        is_clear = 1
        cloud_intensity = 0

    if 'partly' in lcondition:
        is_clear = 1
        cloud_intensity = 1.5

    if 'mostly' in lcondition:
        cloud_intensity = 4

    if 'cloudy' in lcondition:
        cloud_intensity = 5

    if 'fog' in lcondition:
        is_mist = 1
        cloud_intensity = 1.5  # looked at data, seems like having 1.5 here makes it smoother

    if ('haze' in lcondition) or ('mist' in lcondition):
        is_mist = 1

    if ('rain' in lcondition) or ('drizzle' in lcondition) or ('snow' in lcondition) or ('t-storm' in lcondition) or (
            'wintry mix' in lcondition) or ('hail' in lcondition) or ('precipitation' in lcondition):
        cloud_intensity = 5
        is_precip = 1

    if 'snow' in lcondition:
        is_snow = 1

    if ('t-storm' in lcondition) or ('thunder' in lcondition):
        is_thunder = 1

    return is_clear, is_precip, cloud_intensity, is_thunder, is_snow, is_mist
