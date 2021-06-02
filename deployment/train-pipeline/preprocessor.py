import logging
import shutil
import sys
import os
import pandas as pd
import numpy as np
import math

sys.path.insert(1, '../')
import libcommons.libcommons
import config

# ====================================================================================
# ====================================================================================
#	NOAA_preprocessor:
# 		Process NOAA Weather Station Reports into a form more suitable for Machine Learning
# ====================================================================================
# ====================================================================================


#########################################################
def preproc_file(filePath):
    #################################################################################
    # (A) Preliminary Processing
    #################################################################################
    cleaned_df = None
    raw_df = pd.read_csv(filePath, parse_dates=['DATE'])

    # TODO - this added filtering is due to manually discovered NOAA data defects; figure out how to automate
    raw_df = raw_df[raw_df['DATE'] >= '2005-01-01'] # Sky conditions are unavailable b4 2005
    raw_df = raw_df[(raw_df['DATE'] < '2010-08-01') | (raw_df['DATE'] >= '2011-01-01')]  # quincy data set has a hole

    # We want to throw away the SOD reports as they miss Hourly Data:
    raw_df = raw_df[raw_df['REPORT_TYPE'].str.strip() != 'SOD']

    # Get rid of all columns that are not DATE or start with Hourly
    columns_to_drop = list(filter(lambda c: c != 'DATE' and not c.startswith('Hourly'), raw_df.columns.tolist()))
    raw_df = raw_df.drop(columns=columns_to_drop)

    # Put NANs in for variables that are meant to be Numeric but aren't
    numeric_variables = ['HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyWindSpeed', 'HourlyWindDirection',
                         'HourlyDewPointTemperature',
                         'HourlyWindGustSpeed', 'HourlyAltimeterSetting', 'HourlyPressureChange', 'HourlyPrecipitation',
                         'HourlyPressureChange', 'HourlyVisibility']
    raw_df = remove_junk_suffixes_and_prefixes(raw_df, numeric_variables)

    ##################################################################################
    # (B) Custom-process some specific variables
    ##################################################################################

    # Precipitation: replace 'T' ("trace amount") with a small value
    raw_df.loc[raw_df['HourlyPrecipitation'].str.strip() == 'T', 'HourlyPrecipitation'] = 0.005

    # Weather Type: decipher the finicky Government Terminology
    raw_df['HourlyPresentWeatherType'] = raw_df['HourlyPresentWeatherType'].apply(preproc_weather_type)

    # Sky Condition: decipher the finicky Government Terminology
    #	...but first, impute missing values
    raw_df['HourlySkyConditions'] = raw_df['HourlySkyConditions'].fillna(method="ffill")
    raw_df['HourlySkyConditions'] = raw_df['HourlySkyConditions'].fillna(method="bfill")
    raw_df['HourlySkyConditions'], raw_df['HourlyCloudHeight'] = zip(
        *raw_df['HourlySkyConditions'].apply(preproc_sky_conditions))

    # WindDirection: NAN-ify VRB values (aka "Variable Wind")
    raw_df.loc[raw_df['HourlyWindDirection'].str.strip() == 'VRB', 'HourlyWindDirection'] = ""

    # WindGust: replace NANs with 0 (assume no WindGust if not reported)
    raw_df['HourlyWindGustSpeed'] = raw_df['HourlyWindGustSpeed'].fillna(0)

    ####################################################################################
    # (C) Fill in any remaing gaps (NANs) by forward-filling and then back-filling
    ####################################################################################

    raw_df = raw_df.fillna(method="ffill")
    raw_df = raw_df.fillna(method="bfill")

    ################################################################################
    # (D) To resolve duplicates (reports from the same hour), aggregate the tows by hour;
    # 		then, use mean() in most cases; special cases are commented on
    ################################################################################

    # First, convert the relevant quantities to numeric
    raw_df[numeric_variables] = raw_df[numeric_variables].apply(pd.to_numeric)

    # Define the final output
    agg_df = raw_df.groupby(pd.Grouper(freq='1h', key="DATE")).agg(
        Temp=pd.NamedAgg(column="HourlyDryBulbTemperature", aggfunc="mean"),
        Humidity=pd.NamedAgg(column="HourlyRelativeHumidity", aggfunc="mean"),
        DewPoint=pd.NamedAgg(column="HourlyDewPointTemperature", aggfunc="mean"),
        WindSpeed=pd.NamedAgg(column="HourlyWindSpeed", aggfunc="mean"),
        WindGust=pd.NamedAgg(column="HourlyWindGustSpeed", aggfunc="max"),
        # Gusts are gusty- pick the highest to represent the hour
        WindDirection=pd.NamedAgg(column="HourlyWindDirection", aggfunc="mean"),
        Precipitation=pd.NamedAgg(column="HourlyPrecipitation", aggfunc="max"),
        # Precipitation is culumative for the hour
        CloudCondition=pd.NamedAgg(column='HourlySkyConditions', aggfunc=agg_sky_condition),  # Custom Aggregation
        CloudAltitude=pd.NamedAgg(column="HourlyCloudHeight", aggfunc="mean"),
        WeatherType=pd.NamedAgg(column='HourlyPresentWeatherType', aggfunc=agg_weather_type),  # Custom Aggregation
        Pressure=pd.NamedAgg(column="HourlyAltimeterSetting", aggfunc="mean"),
        PressureChange=pd.NamedAgg(column="HourlyPressureChange", aggfunc="mean"),
        Visibility=pd.NamedAgg(column="HourlyVisibility", aggfunc="mean"),
    )

    # Impute any missing values
    agg_df = agg_df.fillna(method="ffill")
    agg_df = agg_df.fillna(method="bfill")

    ################################################################################
    # (E) Derive additional variables helpful for ML purposes
    ################################################################################

    # Summarize Sky Condition in non-categorical format
    agg_df['_is_clear'], agg_df['_cloud_intensity'] = zip(
        *agg_df.apply(lambda row: derive_sky_condition_vars(row['CloudCondition'], row['WeatherType']), axis=1))

    # Summarize Weather Type in non-categorical format
    agg_df['_is_precip'], agg_df['_is_thunder'], agg_df['_is_snow'], agg_df['_is_mist'] = zip(
        *agg_df['WeatherType'].apply(derive_weather_type_vars))

    # Day-of-Year and Hour-of-Day are best encoded as Sin/Cos to reflect their circular nature
    agg_df = agg_df.reset_index()
    agg_df['_day_sin'] = np.sin(2 * np.pi * agg_df['DATE'].dt.dayofyear / 365)
    agg_df['_day_cos'] = np.cos(2 * np.pi * agg_df['DATE'].dt.dayofyear / 365)
    agg_df['_hour_sin'] = np.sin(2 * np.pi * agg_df['DATE'].dt.hour / 24)
    agg_df['_hour_cos'] = np.cos(2 * np.pi * agg_df['DATE'].dt.hour / 24)

    # Wind Direction is given as an angle: also encode that as sine/cosine
    agg_df['_wind_dir_sin'] = np.sin(2 * np.pi * agg_df['WindDirection'] / 360)
    agg_df['_wind_dir_cos'] = np.cos(2 * np.pi * agg_df['WindDirection'] / 360)

    #dump_diagnostics(agg_df)

    return agg_df


#########################################################
def dump_diagnostics(df):
    diag_df = df.groupby('WeatherType').agg(
        AvgPrecip=pd.NamedAgg(column='Precipitation', aggfunc="mean"),
        Count=pd.NamedAgg(column='Precipitation', aggfunc="count")
    )
    print(diag_df.sort_values(by='AvgPrecip'))

    diag_df = df.groupby('_is_precip').agg(
        AvgPrecip=pd.NamedAgg(column='Precipitation', aggfunc="mean"),
        Count=pd.NamedAgg(column='Precipitation', aggfunc="count")
    )
    print(diag_df.sort_values(by='AvgPrecip'))

    diag_df = df.groupby('CloudCondition').agg(
        AvgPrecip=pd.NamedAgg(column='Precipitation', aggfunc="mean"),
        Count=pd.NamedAgg(column='Precipitation', aggfunc="count")
    )
    print(diag_df.sort_values(by='AvgPrecip'))

    diag_df = df.groupby('WeatherType').agg(
        AvgCloudIntensity=pd.NamedAgg(column='_cloud_intensity', aggfunc="mean"),
        Count=pd.NamedAgg(column='_cloud_intensity', aggfunc="count")
    )
    print(diag_df.sort_values(by='AvgCloudIntensity'))


#########################################################

def remove_junk_suffixes_and_prefixes(df, variables):
    for variable in variables:
        try:
            if df[variable].dtypes != np.float64 and df[variable].dtypes != np.int64:
                df.loc[df[variable].str.contains('s', na=False), variable] = ''
                df.loc[df[variable].str.contains('V', na=False), variable] = ''
                df.loc[df[variable].str.contains('\\*', na=False), variable] = ''
        except:
            print("Issue cleaning up variable {}; attempting to proceed".format(variable))
    return df


#########################################################
coverCodes = {
    "CLR": "Clear",
    "FEW": "MostlyClear",
    "SCT": "PartlyCloudy",
    "BKN": "MostlyCloudy",
    "OVC": "Cloudy"
}


def preproc_sky_conditions(skyCondition):
    skyCondition = str(skyCondition).strip()
    if skyCondition == "nan":
        print("TADA sky")
        return None, 0

    # Format Example: BKN:07 6 BKN:07 100 BKN:07 120
    # Sky Condition is reported in 3 layers; as per data set guidance, consider only the 3rd layer to summarize overall condition
    # 	The 3-letter code is the actual condition in Aviation Terminology
    #	The :XX that follows is its numeric representation and can be dropped
    #	The final number is the cloud layer's height above ground, in hundreds of feet

    last_colon_index = skyCondition.rfind(":")
    if last_colon_index == -1:
        return "Clear", 0

    if skyCondition[last_colon_index - 1] == 'V':
        cover = "Obscured"
    else:
        try:
            cover = coverCodes[skyCondition[last_colon_index - 3: last_colon_index]]
        except:
            cover = "Obscured"

    height = 0
    if cover != 'Clear':
        try:
            height = int(skyCondition[last_colon_index + 3:]) * 100
        except ValueError:
            height = 1
    else:
        height = 30000  # If it's clear, assume clouds are very very high

    return cover, height


#########################################################
def agg_sky_condition(series):
    # Err on the side of the most clouded sky
    priorities = ['Cloudy', 'MostlyCloudy', 'PartlyCloudy', 'MostlyClear', 'Clear']
    series = series.values
    for priority in priorities:
        if priority in series:
            return priority

    return 'Obscured'


#########################################################
condition_intensity_map = {
    "Clear": 0,
    "MostlyClear": 1,
    "PartlyCloudy": 2,
    "MostlyCloudy": 4,
    "Cloudy": 5,
    "Obscured": 6
}


def derive_sky_condition_vars(skyCondition, weatherType):
    is_clear = 0
    cloud_intensity = 0

    # We consider the weather "Clear" if sky cover is "PartlyCloudy" or less, AND if it's not foggy/misty
    if (skyCondition in ['Clear', 'MostlyClear', 'PartlyCloudy']) and (
            'Fog' not in weatherType and 'Mist' not in weatherType):
        is_clear = 1

    cloud_intensity = condition_intensity_map[skyCondition]

    return is_clear, cloud_intensity


#########################################################
def preproc_weather_type(weatherType):
    weatherType = str(weatherType).strip()
    thunder_suffix = ""

    if weatherType == "nan":
        return "NoPrecipitation"

    if "TS" in weatherType:
        thunder_suffix = "Thunder"

    if "SN" in weatherType:
        if "-SN" in weatherType:
            return "LightSnow" + thunder_suffix
        elif ("+SN" in weatherType):
            return "HeavySnow" + thunder_suffix
        else:
            return "Snow" + thunder_suffix

    if "RA" in weatherType:
        if "-RA" in weatherType:
            return "LightRain" + thunder_suffix
        elif ("+RA" in weatherType):
            return "HeavyRain" + thunder_suffix
        else:
            return "Rain" + thunder_suffix
    if "DZ" in weatherType:
        return "LightRain"

    if "UP" in weatherType:
        return "UnknownPrecipitation" + thunder_suffix

    if "BR" in weatherType:
        return "Mist" + thunder_suffix

    if "FG" in weatherType:
        return "Fog" + thunder_suffix
    if "HZ" in weatherType or "FU" in weatherType:
        return "Haze" + thunder_suffix
    if "PL" in weatherType or "GR" in weatherType:
        return "Hail" + thunder_suffix
    if "SQ" in weatherType:  # Squall
        return thunder_suffix

    if thunder_suffix != "":
        return thunder_suffix

    return None


def agg_weather_type(series):
    series = series.values
    if len(series) < 1:
        return None

    # Err on the side of the most severe weather
    priorities = ["Heavy", "RainThunder", "Rain", "Snow", "Hail", "UnknownPrecipitation", "Thunder", "Mist", "Fog",
                  "Haze"]
    for priority in priorities:
        for s in series:
            if priority in s:
                return s

    return series[0]


def derive_weather_type_vars(weatherType):
    is_precip = 0
    is_thunder = 0
    is_snow = 0
    is_mist = 0

    if ("Rain" in weatherType) or ("Snow" in weatherType) or ("Hail" in weatherType) or ("Thunder" in weatherType) or (
            "UnknownPrecipitation" in weatherType):
        is_precip = 1

    if "Thunder" in weatherType:
        is_thunder = 1

    if "Snow" in weatherType:
        is_snow = 1

    # According to online resources, mist and fog are different degrees of the same thing: droplets of water suspended in the air
    #	(and neither is considered "precipitation")
    if ("Mist" in weatherType) or ("Fog" in weatherType):
        is_mist = 1

    return is_precip, is_thunder, is_snow, is_mist


#########################################################
def get_preproc_file_name(location, preproc_data_dir):
    return os.path.join(preproc_data_dir, "{}_PREPROC.csv".format(location))


def get_files_to_preprocess(raw_data_dir):
    loc_to_path_dict = {}
    count = 0
    file_list = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))]
    for location in config.Config.LOCATION_CODES:
        loc_matches = [f for f in file_list if location in f]
        loc_to_path_dict[location] = loc_matches
        count = count + len(loc_matches)

    return loc_to_path_dict, count


def preprocess_files(raw_data_dir, preproc_data_dir):
    preproc_count = 0
    files_to_preproc, total_file_cnt = get_files_to_preprocess(raw_data_dir)
    print(files_to_preproc)
    if len (files_to_preproc) > 0 and os.path.exists(preproc_data_dir):
        shutil.rmtree(preproc_data_dir)

    for location in files_to_preproc:
        for file_to_preproc in files_to_preproc[location]:
            preproc_df = preproc_file(file_to_preproc)
            preproc_file_name = get_preproc_file_name(location, preproc_data_dir)

            if not os.path.exists(preproc_file_name):
                os.makedirs(os.path.dirname(preproc_file_name), exist_ok=True)
                preproc_df.to_csv(preproc_file_name, index=True)
            else:
                preproc_df.to_csv(preproc_file_name, mode='a', header=False)
            preproc_count = preproc_count + 1
            print("Preprocess & Saved {} files out of {}, line count = {}".format(preproc_count, total_file_cnt,
                                                                                  len(preproc_df)))


def main(argv):
    print(argv)
    preprocess_files(argv[1], argv[2])


if __name__ == "__main__":
    main(sys.argv[0:])