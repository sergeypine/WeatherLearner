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
    logging.info(current_conditions_df)
    current_conditions_df = pd.DataFrame(current_conditions_df).reset_index()
    current_conditions_df.columns = ['Variable', 'Value']
    logging.info(current_conditions_df)
    current_time = datetime.datetime.now(pytz.timezone(config.Config.TARGET_TIMEZONE))
    current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    return render_template('forecast.html', current_time=current_time,
                           column_names=current_conditions_df.columns.values,
                           row_data=list(current_conditions_df.values.tolist()))


@app.route('/predict_audit')
def predict_audit():
    return {}


@app.route('/model_info')
def model_info():
    return {}

