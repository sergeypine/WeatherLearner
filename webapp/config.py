"""Flask configuration."""


class PredictionTarget():
    def __init__(self, var, lookahead):
        self.var = var
        self.lookahead = lookahead


PREDICTION_TARGET_IS_PRECIP_6H = PredictionTarget('_is_precip', 6)
PREDICTION_TARGET_IS_PRECIP_12H = PredictionTarget('_is_precip', 12)
PREDICTION_TARGET_IS_PRECIP_18H = PredictionTarget('_is_precip', 18)
PREDICTION_TARGET_IS_PRECIP_24H = PredictionTarget('_is_precip', 24)


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
    ALL_PREDICTION_TARGETS = [PREDICTION_TARGET_IS_PRECIP_6H, PREDICTION_TARGET_IS_PRECIP_12H,
                              PREDICTION_TARGET_IS_PRECIP_18H, PREDICTION_TARGET_IS_PRECIP_24H]

    PREDICTION_TARGET_FEATURES = {
        PREDICTION_TARGET_IS_PRECIP_6H: ['_hour_sin', 'DewPoint'],
        PREDICTION_TARGET_IS_PRECIP_12H: ['Precipitation', 'Pressure', 'CloudAltitude', 'WindSpeed', 'WindGust',
                                          '_is_clear', '_is_thunder'],
        PREDICTION_TARGET_IS_PRECIP_18H: ['_day_cos', 'Temp', 'Pressure', 'PressureChange', '_cloud_intensity',
                                          'WindGust', '_wind_dir_sin', '_wind_dir_cos', 'Visibility', '_is_clear',
                                          '_is_thunder'],
        PREDICTION_TARGET_IS_PRECIP_24H: ['_day_cos', 'Temp', 'Humidity', 'Pressure', 'PressureChange',
                                          '_cloud_intensity', 'WindGust', '_wind_dir_sin', '_wind_dir_cos',
                                          'Visibility', '_is_snow', '_is_thunder']
    }

    PREDICTION_TARGET_LOCATIONS = {
        PREDICTION_TARGET_IS_PRECIP_6H: ['Cedar_Rapids', 'Des_Moines', 'Quincy', 'Madison', 'Green_Bay',
                                         'Indianapolis'],
        PREDICTION_TARGET_IS_PRECIP_12H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'Madison', 'St_Louis'],
        PREDICTION_TARGET_IS_PRECIP_18H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Quincy', 'St_Louis', 'Green_Bay'],
        PREDICTION_TARGET_IS_PRECIP_24H: ['Cedar_Rapids', 'Des_Moines', 'Rochester', 'Madison', 'Green_Bay', 'Lansing',
                                          'Indianapolis'],

    }

    PREDICTION_TARGET_LOOKBACKS = {
        PREDICTION_TARGET_IS_PRECIP_6H: 4,
        PREDICTION_TARGET_IS_PRECIP_12H: 24,
        PREDICTION_TARGET_IS_PRECIP_18H: 24,
        PREDICTION_TARGET_IS_PRECIP_24H: 24,
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


class PredictionTarget():
    def __init__(self, var, lookahead):
        self.var = var
        self.lookahead = lookahead
