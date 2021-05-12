"""Flask configuration."""


class PredictionTarget():
    def __init__(self, var, lookahead):
        self.var = var
        self.lookahead = lookahead

    def __repr__(self):
        return "{}+{}hr".format(self.var, self.lookahead)


PREDICTION_TARGET_IS_CLEAR_6H = PredictionTarget('_is_clear', 6)
PREDICTION_TARGET_IS_CLEAR_12H = PredictionTarget('_is_clear', 12)
PREDICTION_TARGET_IS_CLEAR_18H = PredictionTarget('_is_clear', 18)
PREDICTION_TARGET_IS_CLEAR_24H = PredictionTarget('_is_clear', 24)

PREDICTION_TARGET_IS_PRECIP_6H = PredictionTarget('_is_precip', 6)
PREDICTION_TARGET_IS_PRECIP_12H = PredictionTarget('_is_precip', 12)
PREDICTION_TARGET_IS_PRECIP_18H = PredictionTarget('_is_precip', 18)
PREDICTION_TARGET_IS_PRECIP_24H = PredictionTarget('_is_precip', 24)

PREDICTION_TARGET_TEMP_6H = PredictionTarget('Temp', 6)
PREDICTION_TARGET_TEMP_12H = PredictionTarget('Temp', 12)
PREDICTION_TARGET_TEMP_18H = PredictionTarget('Temp', 18)
PREDICTION_TARGET_TEMP_24H = PredictionTarget('Temp', 24)

PREDICTION_TARGET_WINDSPEED_6H = PredictionTarget('WindSpeed', 6)
PREDICTION_TARGET_WINDSPEED_12H = PredictionTarget('WindSpeed', 12)
PREDICTION_TARGET_WINDSPEED_18H = PredictionTarget('WindSpeed', 18)
PREDICTION_TARGET_WINDSPEED_24H = PredictionTarget('WindSpeed', 24)


class Config(object):
    # =========================================================
    LOCATION_CODES = {
        'Chicago': 'KMDW',
        'Cedar_Rapids': 'KCID',
        'Des_Moines': 'KDSM',
        'Madison': 'KMSN',
        'Quincy': 'KUIN',
        'St_Louis': 'KSTL',
        'Rochester': 'KRST',
        'Green_Bay': 'KGRB',
        'Lansing': 'KLAN',
        'Indianapolis': 'KIND',
        'Columbus': 'KCMH',
        'Toledo': 'KTOL'
    }
    TARGET_LOCATION = 'Chicago'
    TARGET_TIMEZONE = 'America/Chicago'
    MAX_LOOK_BACK_HOURS = 24
    # =========================================================
    ALL_PREDICTION_TARGETS = [PREDICTION_TARGET_IS_CLEAR_6H, PREDICTION_TARGET_IS_CLEAR_12H,
                              PREDICTION_TARGET_IS_CLEAR_18H, PREDICTION_TARGET_IS_CLEAR_24H,

                              PREDICTION_TARGET_IS_PRECIP_6H, PREDICTION_TARGET_IS_PRECIP_12H,
                              PREDICTION_TARGET_IS_PRECIP_18H, PREDICTION_TARGET_IS_PRECIP_24H,

                              PREDICTION_TARGET_TEMP_6H, PREDICTION_TARGET_TEMP_12H,
                              PREDICTION_TARGET_TEMP_18H, PREDICTION_TARGET_TEMP_24H,

                              PREDICTION_TARGET_WINDSPEED_6H, PREDICTION_TARGET_WINDSPEED_12H,
                              PREDICTION_TARGET_WINDSPEED_18H, PREDICTION_TARGET_WINDSPEED_24H]

    PREDICTION_TARGET_FEATURES = {
        PREDICTION_TARGET_IS_CLEAR_6H: ['_day_cos', 'DewPoint', '_cloud_intensity', 'CloudAltitude', '_is_thunder'],
        PREDICTION_TARGET_IS_CLEAR_12H: ['_day_cos', 'Temp', 'PressureChange', '_cloud_intensity', 'CloudAltitude',
                                         '_wind_dir_sin', '_is_precip'],
        PREDICTION_TARGET_IS_CLEAR_18H: ['_day_cos', '_hour_cos', 'Temp', 'Pressure', 'PressureChange', 'WindSpeed',
                                         '_wind_dir_sin'],
        PREDICTION_TARGET_IS_CLEAR_24H: ['_day_cos', 'Temp', 'Pressure', 'PressureChange'],

        PREDICTION_TARGET_IS_PRECIP_6H: ['_hour_sin', 'DewPoint'],
        PREDICTION_TARGET_IS_PRECIP_12H: ['Precipitation', 'Pressure', 'CloudAltitude', 'WindSpeed', 'WindGust',
                                          '_is_clear', '_is_thunder'],
        PREDICTION_TARGET_IS_PRECIP_18H: ['_day_cos', 'Temp', 'Pressure', 'PressureChange', '_cloud_intensity',
                                          'WindGust', '_wind_dir_sin', '_wind_dir_cos', 'Visibility', '_is_clear',
                                          '_is_thunder'],
        PREDICTION_TARGET_IS_PRECIP_24H: ['_day_cos', 'Temp', 'Humidity', 'Pressure', 'PressureChange',
                                          '_cloud_intensity', 'WindGust', '_wind_dir_sin', '_wind_dir_cos',
                                          'Visibility', '_is_snow', '_is_thunder'],

        PREDICTION_TARGET_TEMP_6H: ['_hour_sin', 'DewPoint', 'Humidity', 'PressureChange', '_cloud_intensity',
                                    '_is_thunder'],
        PREDICTION_TARGET_TEMP_12H: ['_hour_sin', 'Precipitation', 'PressureChange', '_cloud_intensity', 'WindGust'],
        PREDICTION_TARGET_TEMP_18H: ['_hour_sin', 'Precipitation', 'PressureChange', '_cloud_intensity', 'WindGust',
                                     '_wind_dir_sin', '_is_thunder'],
        PREDICTION_TARGET_TEMP_24H: ['_hour_sin', 'DewPoint', 'Precipitation', 'Pressure', '_cloud_intensity', 'WindSpeed',
                                     '_wind_dir_cos'],

        PREDICTION_TARGET_WINDSPEED_6H: ['_hour_sin', '_hour_cos', 'Humidity', 'Pressure', 'PressureChange'],
        PREDICTION_TARGET_WINDSPEED_12H: ['_day_cos', '_hour_sin', '_hour_cos', 'DewPoint', 'Humidity', 'Pressure',
                                          'PressureChange', 'CloudAltitude', 'WindGust', '_wind_dir_sin',
                                          '_wind_dir_cos'],
        PREDICTION_TARGET_WINDSPEED_18H: ['_day_cos', '_hour_sin', 'Temp', 'PressureChange', '_cloud_intensity',
                                          'CloudAltitude'],
        PREDICTION_TARGET_WINDSPEED_24H: ['_day_cos', '_hour_sin', 'Humidity', 'Precipitation', 'PressureChange',
                                          'WindGust', '_wind_dir_cos', '_is_thunder'],
    }

    PREDICTION_TARGET_LOCATIONS = {
        PREDICTION_TARGET_IS_CLEAR_6H: ['Cedar_Rapids', 'Rochester', 'Madison', 'St_Louis', 'Green_Bay'],
        PREDICTION_TARGET_IS_CLEAR_12H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'Madison',
                                         'St_Louis', 'Green_Bay'],
        PREDICTION_TARGET_IS_CLEAR_18H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'Columbus'],
        PREDICTION_TARGET_IS_CLEAR_24H:  ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Madison', 'St_Louis', 'Green_Bay',
                                          'Lansing', 'Indianapolis'],

        PREDICTION_TARGET_IS_PRECIP_6H: ['Cedar_Rapids', 'Des_Moines', 'Quincy', 'Madison', 'Green_Bay',
                                         'Indianapolis'],
        PREDICTION_TARGET_IS_PRECIP_12H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'Madison', 'St_Louis'],
        PREDICTION_TARGET_IS_PRECIP_18H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'St_Louis', 'Green_Bay'],
        PREDICTION_TARGET_IS_PRECIP_24H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Madison', 'Green_Bay', 'Lansing',
                                          'Indianapolis'],

        PREDICTION_TARGET_TEMP_6H: ['Cedar_Rapids', 'Des_Moines', 'Rochester'],
        PREDICTION_TARGET_TEMP_12H:  ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'Madison', 'Lansing',
                                      'Indianapolis'],
        PREDICTION_TARGET_TEMP_18H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy'],
        PREDICTION_TARGET_TEMP_24H: ['Cedar_Rapids', 'Des_Moines', 'St_Louis', 'Indianapolis'],

        PREDICTION_TARGET_WINDSPEED_6H: ['Cedar_Rapids', 'Des_Moines', 'Quincy', 'Madison', 'St_Louis', 'Lansing',
                                         'Indianapolis'],
        PREDICTION_TARGET_WINDSPEED_12H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'Indianapolis'],
        PREDICTION_TARGET_WINDSPEED_18H: ['Cedar_Rapids', 'Des_Moines', 'Madison', 'St_Louis'],
        PREDICTION_TARGET_WINDSPEED_24H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy']
    }

    PREDICTION_TARGET_LOOKBACKS = {
        PREDICTION_TARGET_IS_CLEAR_6H: 4,
        PREDICTION_TARGET_IS_CLEAR_12H: 4,
        PREDICTION_TARGET_IS_CLEAR_18H: 4,
        PREDICTION_TARGET_IS_CLEAR_24H: 4,

        PREDICTION_TARGET_IS_PRECIP_6H: 4,
        PREDICTION_TARGET_IS_PRECIP_12H: 24,
        PREDICTION_TARGET_IS_PRECIP_18H: 24,
        PREDICTION_TARGET_IS_PRECIP_24H: 24,

        PREDICTION_TARGET_TEMP_6H: 4,
        PREDICTION_TARGET_TEMP_12H: 4,
        PREDICTION_TARGET_TEMP_18H: 4,
        PREDICTION_TARGET_TEMP_24H: 4,

        PREDICTION_TARGET_WINDSPEED_6H: 4,
        PREDICTION_TARGET_WINDSPEED_12H: 4,
        PREDICTION_TARGET_WINDSPEED_18H: 4,
        PREDICTION_TARGET_WINDSPEED_24H: 4,
    }

    # Aggregation Half-Intervals for the supported predicted variables
    PREDICTED_VARIABLE_AHI = {
        '_is_precip': 3,
        '_is_clear': 3,
        'Temp': 1,
        'WindSpeed': 2,
    }

    PREDICTED_VARIABLE_AGG_RULES = {
        '_is_precip': 'ANY',
        '_is_clear': 'ALL',
        'Temp': 'AVG',
        'WindSpeed': 'AVG',
    }
    # =========================================================
    LOCATION_DATASET_FILE_FORMAT = '../processed-data/noaa_2011-2020_{}_PREPROC.csv'

    PREDICTION_TARGET_MODEL_TYPES = {
        PREDICTION_TARGET_IS_CLEAR_6H: 'LINEAR',
        PREDICTION_TARGET_IS_CLEAR_12H: 'LINEAR',
        PREDICTION_TARGET_IS_CLEAR_18H: 'NN',
        PREDICTION_TARGET_IS_CLEAR_24H: 'NN',

        PREDICTION_TARGET_IS_PRECIP_6H: 'LINEAR',
        PREDICTION_TARGET_IS_PRECIP_12H: 'CNN',
        PREDICTION_TARGET_IS_PRECIP_18H: 'CNN',
        PREDICTION_TARGET_IS_PRECIP_24H: 'CNN',

        PREDICTION_TARGET_TEMP_6H: 'DNN',
        PREDICTION_TARGET_TEMP_12H: 'NN',
        PREDICTION_TARGET_TEMP_18H: 'DNN',
        PREDICTION_TARGET_TEMP_24H: 'LINEAR',

        PREDICTION_TARGET_WINDSPEED_6H: 'NN',
        PREDICTION_TARGET_WINDSPEED_12H: 'NN',
        PREDICTION_TARGET_WINDSPEED_18H: 'NN',
        PREDICTION_TARGET_WINDSPEED_24H: 'LINEAR'
    }


class PredictionTarget():
    def __init__(self, var, lookahead):
        self.var = var
        self.lookahead = lookahead
