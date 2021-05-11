from flask import Flask

import config
from logging.config import dictConfig
import reading_retriever
import predictor
import trainer

# ===============================================================
app = Flask(__name__)
if __name__ == '__main__':
    app.run()
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
@app.route('/')
def home():
    # reading_retriever.retrieve_latest_readings()
    predictor.generate_predictions()
    return "Hello World"


@app.route('/train')
def train():
    trainer.train_models()
    return "Goodbye World"
