import sys
import time
import schedule
import logging

sys.path.insert(1, '../')
import config


def generate_forecast_job():
    logging.info("Start Forecast Job")
    logging.info("End Forecast Job")


def update_prediction_audit_job():
    logging.info("Start Backfill Job")
    logging.info("End Backfill Job")


def main():
    logging.basicConfig(filename=conf.DATA_SERVICE_LOG_FILE,
                        format=conf.DATA_SERVICE_LOG_FORMAT,
                        level=conf.DATA_SERVICE_LOG_LEVEL)

    # Pre-execute the jobs
    generate_forecast_job()
    update_prediction_audit_job()

    # Keep running the jobs on a schedule
    schedule.every(conf.DATA_SERVICE_FORECAST_INTERVAL_MINUTES).minutes.do(generate_forecast_job)
    schedule.every(conf.DATA_SERVICE_BACKFILL_INTERVAL_MINUTES).minutes.do(update_prediction_audit_job)
    while 1:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    conf = config.Config
    main()
