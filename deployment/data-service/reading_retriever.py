import sys
import logging
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

sys.path.insert(1, '../')
import config
import libcommons.libcommons


class ReadingRetriever():
    def __init__(self):

        self.PAGE_LOAD_RETRIES = 4
        self.WAIT_TIME = 2

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_prefs = {}
        chrome_options.experimental_options["prefs"] = chrome_prefs
        chrome_prefs["profile.default_content_settings"] = {"images": 2}
        self._driver = webdriver.Chrome(executable_path="bin/chromedriver",
                                        chrome_options=chrome_options)
        self.conf = config.Config
        self.data_store = libcommons.libcommons.DataStore()

    def retrieve_for_date_and_location(self, date, location):
        logging.info("Retrieving readings for {}".format(location))

        location_code = self.conf.LOCATION_CODES[location]

        html = self.get_html_for_date(date, location_code)
        data = self.get_hourly_for_date_from_html(html, date)

        column_names = ['Time', 'Temp', 'DewPoint', 'Humidity', 'Wind', 'WindSpeed', 'WindGust', 'Pressure', 'Precip',
                        'Condition']
        data_df = pd.DataFrame(data, columns=column_names)
        loc_data_noaa = self.loc_data_to_noo_format(self, data_df)

        self.data_store.readings_append(location, loc_data_noaa)
        logging.info("Finished retrieving readings for location {}; {} rows found".format(location,
                                                                                          len(loc_data_noaa)))

    def get_html_for_date(self, date, location_code):
        # URL example: https://www.wunderground.com/history/daily/KMDW/date/2015-1-21
        target_url = 'https://www.wunderground.com/history/daily/{}/date/{}'.format(location_code, date)

        attempts = 0

        while attempts < self.PAGE_LOAD_RETRIES:
            try:
                self._driver.get(target_url)
                time.sleep(self.WAIT_TIME)  # blegh...

                soup = BeautifulSoup(self._driver.page_source, 'lxml')
                table_html = soup.find_all('table')[1]
                return table_html
            except:
                attempts = attempts + 1
                if attempts >= self.PAGE_LOAD_RETRIES:
                    logging.info("Unexpected error loading page: {} {}".format(sys.exc_info()[0],
                                                                               sys.exc_info()[1]))

        return None

    @staticmethod
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

    @staticmethod
    def loc_data_to_noo_format(self, loc_data: pd.DataFrame):
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
        loc_data['Precipitation'] = loc_data['Precip'].str.replace(' in', '')

        # Transform wind direction into 3 components: Northerly, Easterly and is_var
        loc_data['_wind_dir_sin'], loc_data['_wind_dir_cos'] = zip(
            *loc_data['Wind'].apply(self.get_wind_dir_sin_cos))

        # Transform condition into numeric components
        loc_data['_is_clear'], loc_data['_is_precip'], loc_data['_cloud_intensity'], loc_data['_is_thunder'], \
        loc_data['_is_snow'], loc_data['_is_mist'] = zip(*loc_data['Condition'].apply(self.get_condition_components))

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
        cleaned_df = cleaned_df.sort_values(by='DATE').drop(columns=['Time'])

        # Resample so there are exactly 24 hourly readings;
        #  (drop duplicate hours, then add missing hours and impune)
        cleaned_df = self.df_to_24h_freq(cleaned_df)

        # Create day-of-year and hour-of-day sin/cos columns to capture the cyclical nature of time
        cleaned_df['_day_sin'] = np.sin(2 * np.pi * cleaned_df['DATE'].dt.dayofyear / 365)
        cleaned_df['_day_cos'] = np.cos(2 * np.pi * cleaned_df['DATE'].dt.dayofyear / 365)

        cleaned_df['_hour_sin'] = np.sin(2 * np.pi * cleaned_df['DATE'].dt.hour / 24)
        cleaned_df['_hour_cos'] = np.cos(2 * np.pi * cleaned_df['DATE'].dt.hour / 24)

        # Shed the unnecessary columns
        cleaned_df = cleaned_df.drop(columns=['Condition', 'Wind', 'Precip'])
        return cleaned_df

    @staticmethod
    def df_to_24h_freq(df):
        df = df.drop_duplicates(subset=['DATE']).reindex()

        frame_date = df['DATE'].iloc[2].date() # Head & Tail entries are sometimes from other dates
        ts_index = pd.date_range(start=frame_date, periods=24, freq='H')
        resampled_df = pd.DataFrame(ts_index, columns=['DATE']).merge(df, on=['DATE'], how='outer')

        resampled_df = resampled_df[resampled_df['DATE'].dt.date == frame_date] # Chuck extraneous dates

        resampled_df = resampled_df.fillna(method='ffill')
        resampled_df = resampled_df.fillna(method='bfill')
        return resampled_df

    @staticmethod
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

    @staticmethod
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
        elif 'partly' in lcondition:
            is_clear = 1
            cloud_intensity = 1.5
        elif 'mostly' in lcondition:
            cloud_intensity = 4
        elif 'cloudy' in lcondition:
            cloud_intensity = 5

        if 'fog' in lcondition:
            is_mist = 1
            cloud_intensity = 1.5  # looked at data, seems like having 1.5 here makes it smoother

        if ('haze' in lcondition) or ('mist' in lcondition):
            is_mist = 1

        if ('rain' in lcondition) or ('drizzle' in lcondition) or ('snow' in lcondition) or (
                't-storm' in lcondition) or (
                'wintry mix' in lcondition) or ('hail' in lcondition) or ('precipitation' in lcondition):
            cloud_intensity = 5
            is_precip = 1

        if 'snow' in lcondition:
            is_snow = 1

        if ('t-storm' in lcondition) or ('thunder' in lcondition):
            is_thunder = 1

        return is_clear, is_precip, cloud_intensity, is_thunder, is_snow, is_mist
