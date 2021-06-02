import sys
import logging
from logging.config import dictConfig
import pandas as pd
from flask import Flask
from flask import render_template
import numpy as np
import datetime
import pytz
import json
sys.path.insert(1, '../')
import config
import libcommons.libcommons
import webapp_utils

app = Flask(__name__)
if __name__ == '__main__':
    app.run(host='0.0.0.0')

# ===============================================================
app.config.from_object(config.Config())
logging.basicConfig(filename=config.Config.WEBAPP_LOG_FILE,
                    format=config.Config.WEBAPP_LOG_FORMAT,
                    level=config.Config.WEBAPP_LOG_LEVEL)


# ===========================================================

@app.route('/forecast')
def forecast():
    """Primary route that returns latest forecast"""
    logging.info("Received /forecast request")
    forecast_df = webapp_utils.get_forecast_df()
    current_conditions_df = webapp_utils.get_current_conditions_df()

    current_time = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE))
    current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

    if forecast_df is not None and len(forecast_df) > 0 and \
            current_conditions_df is not None and len(current_conditions_df) > 0:

        current_conditions_df = webapp_utils.format_forecast(current_conditions_df)
        forecast_df = webapp_utils.format_forecast(forecast_df)
        table_info = [
            {'title': 'Last Known Conditions',
             'column_names': current_conditions_df.columns.values,
             'row_data': list(current_conditions_df.values.tolist())},
            {'title': 'Current Forecast',
             'column_names': forecast_df.columns.values,
             'row_data': list(forecast_df.values.tolist())},
        ]
        return render_template('forecast.html', current_time=current_time, table_info=table_info)
    else:
        return render_template('forecast_nodata.html', current_time=current_time)


@app.route('/predict_audit')
def predict_audit():
    logging.info("Received /predict_audit request")

    # TODO - consider caching in app context
    temp_audit = libcommons.libcommons.DataStore().prediction_audit_load('Temp')
    wind_audit = libcommons.libcommons.DataStore().prediction_audit_load('WindSpeed')
    _is_clear_audit = libcommons.libcommons.DataStore().prediction_audit_load('_is_clear')
    _is_precip_audit = libcommons.libcommons.DataStore().prediction_audit_load('_is_precip')

    data_present = all(audit is not None and len(audit) > 0
                       for audit in [temp_audit, wind_audit, _is_clear_audit, _is_precip_audit])

    if data_present:
        #Get rid of NANs
        for audit in [temp_audit, wind_audit, _is_clear_audit, _is_precip_audit]:
            audit.dropna(inplace=True)

        # Do not include every hour, that is overwhelming!
        temp_audit = webapp_utils.reduce_audit_granurality(temp_audit)
        wind_audit = webapp_utils.reduce_audit_granurality(wind_audit)
        _is_clear_audit = webapp_utils.reduce_audit_granurality(_is_clear_audit)
        _is_precip_audit = webapp_utils.reduce_audit_granurality(_is_precip_audit)

        # Change 0's and 1's no NO's and YES's
        _is_clear_audit = webapp_utils.format_yesno_tbl(_is_clear_audit)
        _is_precip_audit = webapp_utils.format_yesno_tbl(_is_precip_audit)

        # Convert floats to ints
        temp_audit = webapp_utils.format_numeric_tbl(temp_audit)
        wind_audit = webapp_utils.format_numeric_tbl(wind_audit)

        # Drop minutes
        for audit in [temp_audit, wind_audit, _is_clear_audit, _is_precip_audit]:
            audit['DATE'] = audit['DATE'].astype("datetime64").dt.strftime("%m-%d-%y %H:%M")

        table_info = [
            {'title': 'Temperature',
             'column_names': temp_audit.columns.values,
             'row_data': list(temp_audit.values.tolist())},
            {'title': 'Wind',
             'column_names': wind_audit.columns.values,
             'row_data': list(wind_audit.values.tolist())},
            {'title': 'Clear Sky',
             'column_names': _is_clear_audit.columns.values,
             'row_data': list(_is_clear_audit.values.tolist())},
            {'title': 'Precipitation',
             'column_names': _is_precip_audit.columns.values,
             'row_data': list(_is_precip_audit.values.tolist())},
        ]
        return render_template("audit.html", table_info=table_info)
    else:
        return render_template("audit_nodata.html")


@app.route('/model_info')
def model_info():
    logging.info("Received /model_info request")
    with open(libcommons.libcommons.get_model_info_file()) as json_file:
        model_info_dict = json.load(json_file)
        classification_info, regression_info = [model_info_dict['CLASSIFICATION'], model_info_dict['REGRESSION']]

        regression_tbl, classification_tbl =  webapp_utils.format_model_info(regression_info, classification_info)

        table_info = [
            {'title': 'Regression Models',
             'column_names': regression_tbl.columns.values,
             'row_data': list(regression_tbl.values.tolist())},
            {'title': 'Classification Models',
             'column_names': classification_tbl.columns.values,
             'row_data': list(classification_tbl.values.tolist())}
        ]

        return render_template("model_info.html", table_info=table_info)



