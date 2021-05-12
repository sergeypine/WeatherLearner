import numpy as np
import pandas as pd
from flask import Flask
import config
from logging.config import dictConfig
import reading_retriever
import predictor
import forecast_commons
import trainer

app = Flask(__name__)

# ===============================================================
app.config.from_object(config.Config())
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


# ===========================================================
@app.route('/refresh')
def update_predictions():
    """Regularly scheduled (e.g. Crontab) to get latest prediction data"""
    # Save latest weather readings into files
    reading_retriever.retrieve_latest_readings()

    # Use those readings to generate predictions and also save them to a file
    predictor.generate_predictions()

    return "OK"


@app.route('/forecast')
def home():
    """Main method to return the latest forecast"""
    df = pd.read_csv(forecast_commons.get_latest_predictions_file())

    # TODO - this is for demo only, redo using templates etc
    df['Val'] = df['Val'].astype(int)
    df['Val'] = np.where(df['Var'] == 'Temp', df['Val'].astype(str) + ' F', df['Val'])
    df['Val'] = np.where(df['Var'] == 'WindSpeed', df['Val'].astype(str) + ' mph', df['Val'])
    df.loc[((df['Var'] == '_is_precip') | (df['Var'] == '_is_clear')) & (df['Val'] == 1), 'Val'] = 'Yes'
    df.loc[((df['Var'] == '_is_precip') | (df['Var'] == '_is_clear')) & (df['Val'] == 0), 'Val'] = 'No'

    new_dict = {}
    timestamps = df['Timestamp'].unique()
    variables = df['Var'].unique()
    new_columns = ['Timestamp']
    for variable in variables:
        new_columns.append(variable)
    for ts in timestamps:
        new_dict[ts] = [ts]
        for variable in variables:
            target = df[(df['Timestamp'] == ts) & (df['Var'] == variable)]
            new_dict[ts].append(target.iloc[0]['Val'])

    new_df = pd.DataFrame.from_dict(new_dict, columns=new_columns, orient='index')
    return new_df.to_html(index=False)


@app.route('/train')
def train():
    """Backdoor method to trigger complete model training"""
    trainer.train_models()
    return "Goodbye World"


# ========================================================
if __name__ == "__main__":
    app.run()
