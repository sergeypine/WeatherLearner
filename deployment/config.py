"""App configuration."""


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

PREDICTION_TARGET_IS_SNOW_6H = PredictionTarget('_is_snow', 6)
PREDICTION_TARGET_IS_SNOW_12H = PredictionTarget('_is_snow', 12)
PREDICTION_TARGET_IS_SNOW_18H = PredictionTarget('_is_snow', 18)


class Config(object):
    # =========================================================
    DATA_SERVICE_FORECAST_INTERVAL_MINUTES = 15
    DATA_SERVICE_BACKFILL_INTERVAL_MINUTES = 2
    DATA_SERVICE_BACKFILL_INTERVAL_DAYS = 60
    DATA_SERVICE_BACKFILL_BATCH_SIZE = 2
    DATA_SERVICE_NUM_RETRIEVER_WORKERS = 4
    DATA_SERVICE_LOG_FILE = '../logs/data_service.log'
    DATA_SERVICE_LOG_LEVEL = 'INFO'
    DATA_SERVICE_LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'

    WEBAPP_LOG_FILE = '../logs/webapp.log'
    WEBAPP_LOG_LEVEL = 'INFO'
    WEBAPP_LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'

    WEBAPP_MAX_READING_DELAY_HOURS = 3
    WEBAPP_HRS_INCLUDED_AUDIT = [2, 8, 14, 20]

    TRAINER_LOG_FILE = "training.log"
    TRAINER_LOG_LEVEL = 'INFO'
    TRAINER_LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
    # =========================================================
    DATA_STORE_BASE_DIR = "../data_store"
    MODELS_BASE_DIR = "../pretrained"
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
        'Indianapolis': 'KIND'
    }
    TARGET_LOCATION = 'Chicago'
    TARGET_TIMEZONE = 'America/Chicago'
    MAX_LOOK_BACK_HOURS = 24
    # =========================================================
    ALL_PREDICTION_TARGETS = [PREDICTION_TARGET_IS_PRECIP_12H]

    TEMP_FEATS = ['_day_cos', '_day_sin', '_hour_sin', '_hour_cos', 'DewPoint', 'Precipitation', 'Pressure', '_cloud_intensity', 'WindSpeed', '_wind_dir_sin', '_wind_dir_cos', '_is_thunder']
    WINDSPEED_FEATS = ['_day_cos', '_day_sin', '_hour_sin', '_hour_cos', 'Temp', 'DewPoint', 'Humidity', 'Pressure', 'Precipitation', 'WindGust', '_wind_dir_sin', '_wind_dir_cos', '_is_thunder']

    IS_CLEAR_FEATS = ['_day_cos', '_day_sin', 'Temp', 'Pressure', 'Humidity', 'WindSpeed', '_wind_dir_sin', '_wind_dir_cos', '_cloud_intensity', '_is_precip']

    IS_PRECIP_FEATS = ['_day_cos', '_day_sin', '_hour_cos', '_hour_sin', 'Temp', 'Pressure', 'DewPoint', 'Humidity', 'WindSpeed', 'WindGust', '_wind_dir_sin', '_wind_dir_cos', '_cloud_intensity', '_is_thunder', '_is_snow']
    IS_SNOW_FEATS = ['_day_cos', '_day_sin', '_hour_cos', '_hour_sin', 'Temp', 'Pressure', 'DewPoint', 'Humidity', 'WindSpeed', 'WindGust', '_wind_dir_sin', '_wind_dir_cos', '_cloud_intensity', '_is_precip', '_is_thunder']

    PREDICTION_TARGET_FEATURES = {
        PREDICTION_TARGET_IS_CLEAR_6H: IS_CLEAR_FEATS,
        PREDICTION_TARGET_IS_CLEAR_12H: IS_CLEAR_FEATS,
        PREDICTION_TARGET_IS_CLEAR_18H: IS_CLEAR_FEATS,
        PREDICTION_TARGET_IS_CLEAR_24H: IS_CLEAR_FEATS,

        PREDICTION_TARGET_IS_PRECIP_6H: IS_PRECIP_FEATS,
        PREDICTION_TARGET_IS_PRECIP_12H: IS_PRECIP_FEATS,
        PREDICTION_TARGET_IS_PRECIP_18H: IS_PRECIP_FEATS,
        PREDICTION_TARGET_IS_PRECIP_24H: IS_PRECIP_FEATS,

        PREDICTION_TARGET_TEMP_6H: TEMP_FEATS,
        PREDICTION_TARGET_TEMP_12H: TEMP_FEATS,
        PREDICTION_TARGET_TEMP_18H: TEMP_FEATS,
        PREDICTION_TARGET_TEMP_24H: TEMP_FEATS,

        PREDICTION_TARGET_WINDSPEED_6H: WINDSPEED_FEATS,
        PREDICTION_TARGET_WINDSPEED_12H: WINDSPEED_FEATS,
        PREDICTION_TARGET_WINDSPEED_18H: WINDSPEED_FEATS,
        PREDICTION_TARGET_WINDSPEED_24H: WINDSPEED_FEATS,

        PREDICTION_TARGET_IS_SNOW_6H: IS_SNOW_FEATS,
        PREDICTION_TARGET_IS_SNOW_12H: IS_SNOW_FEATS,
        PREDICTION_TARGET_IS_SNOW_18H: IS_SNOW_FEATS,
    }

    LOCATIONS = ['Cedar_Rapids', 'Des_Moines', 'Madison', 'Rochester', 'Quincy', 'St_Louis', 'Green_Bay', 'Indianapolis', 'Lansing']
    PREDICTION_TARGET_LOCATIONS = {
        PREDICTION_TARGET_IS_CLEAR_6H: LOCATIONS,
        PREDICTION_TARGET_IS_CLEAR_12H: LOCATIONS,
        PREDICTION_TARGET_IS_CLEAR_18H: LOCATIONS,
        PREDICTION_TARGET_IS_CLEAR_24H: LOCATIONS,

        PREDICTION_TARGET_IS_PRECIP_6H: LOCATIONS,
        PREDICTION_TARGET_IS_PRECIP_12H: LOCATIONS,
        PREDICTION_TARGET_IS_PRECIP_18H: LOCATIONS,
        PREDICTION_TARGET_IS_PRECIP_24H: LOCATIONS,

        PREDICTION_TARGET_TEMP_6H: LOCATIONS,
        PREDICTION_TARGET_TEMP_12H: LOCATIONS,
        PREDICTION_TARGET_TEMP_18H: LOCATIONS,
        PREDICTION_TARGET_TEMP_24H: LOCATIONS,

        PREDICTION_TARGET_WINDSPEED_6H: LOCATIONS,
        PREDICTION_TARGET_WINDSPEED_12H: LOCATIONS,
        PREDICTION_TARGET_WINDSPEED_18H: LOCATIONS,
        PREDICTION_TARGET_WINDSPEED_24H: LOCATIONS,

        PREDICTION_TARGET_IS_SNOW_6H: LOCATIONS,
        PREDICTION_TARGET_IS_SNOW_12H: LOCATIONS,
        PREDICTION_TARGET_IS_SNOW_18H: LOCATIONS
    }

    PREDICTION_TARGET_LOOKBACKS = {
        PREDICTION_TARGET_IS_CLEAR_6H: 4,
        PREDICTION_TARGET_IS_CLEAR_12H: 4,
        PREDICTION_TARGET_IS_CLEAR_18H: 4,
        PREDICTION_TARGET_IS_CLEAR_24H: 4,

        PREDICTION_TARGET_IS_PRECIP_6H: 4,
        PREDICTION_TARGET_IS_PRECIP_12H: 36,
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

        PREDICTION_TARGET_IS_SNOW_6H: 36,
        PREDICTION_TARGET_IS_SNOW_12H: 36,
        PREDICTION_TARGET_IS_SNOW_18H: 36
    }

    # Aggregation Half-Intervals for the supported predicted variables
    PREDICTED_VARIABLE_AHI = {
        '_is_precip': 3,
        '_is_clear': 3,
        'Temp': 1,
        'WindSpeed': 2,

        '_is_snow': 3,
    }

    PREDICTED_VARIABLE_AGG_RULES = {
        '_is_precip': 'ANY',
        '_is_clear': 'ALL',
        'Temp': 'AVG',
        'WindSpeed': 'AVG',

        '_is_snow': 'ANY',
    }
    # =========================================================

    PREDICTION_TARGET_MODEL_TYPES = {
        PREDICTION_TARGET_IS_PRECIP_12H: 'DCNN'
    }
