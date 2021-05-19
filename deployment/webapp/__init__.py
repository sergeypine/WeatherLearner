# Entrypoint into the User Facing Webapp
from flask import Flask


@app.route('/forecast')
def forecast():
    return {}


@app.route('/predict_audit')
def predict_audit():
    return {}


@app.route('/model_info')
def model_info():
    return {}

