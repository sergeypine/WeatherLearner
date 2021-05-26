# Entrypoint into the User Facing Webapp
import sys
import logging
from logging.config import dictConfig
import pandas as pd
from flask import Flask
from flask import render_template
import datetime
import pytz
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
    return render_template("audit.html")


@app.route('/model_info')
def model_info():
    return {}

