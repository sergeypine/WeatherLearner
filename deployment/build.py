import sys
import os
import subprocess

# ========== Get command line arguments
do_train = False
do_test = True
do_build = True
do_deploy_locally = True
do_deploy_remotely = False

cur_dir = os.getcwd()

if do_train:
    # TODO

    # ========== Pre-process NOAA files
    # TODO

    # ========== (Optional) Select Features
    # TODO
    pass

#  ========== Train Models
# TODO

#  ==========  Run Tests
# NOTE: requires ### export PYTHONPATH=. pytest ###
if do_test:
    os.chdir('webapp')
    subprocess.run(["pytest"], check=True).check_returncode()

    os.chdir('../libcommons')
    subprocess.run(["pytest"], check=True).check_returncode()

    os.chdir('../data-service')
    subprocess.run(["pytest"], check=True).check_returncode()

#  ========== (Optional) Pull Weather Readings
# TODO

#  ========== Build Docker
if do_build:
    os.chdir(cur_dir)
    subprocess.run(["docker", "build", "-t", "weather-predictor:latest", "."], check=True).check_returncode()

#  ========== Deploy
if do_deploy_locally:
    try:
        subprocess.run(["docker", "stop",  "weather-predictor"], check=True).check_returncode()
    except:
        print("Container was not running, OK to proceed")
    try:
        subprocess.run(["docker", "rm", "-f", "weather-predictor"], check=True).check_returncode()
    except:
        print("Container did not exist, OK to proceed")

    subprocess.Popen(["docker", "run", "--name", "weather-predictor", "-v$PWD.", "-p", "5000:5000", "weather-predictor:latest"], close_fds=True)

if do_deploy_remotely:
    # TODO
    pass