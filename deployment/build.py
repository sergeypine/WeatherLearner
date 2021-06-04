import sys
import os
import subprocess
import shutil
import json

# Sensible defaults for which steps we'll do
do_train = True
do_preprocess = True
do_test = True
do_build = True
do_deploy_locally = True
do_deploy_remotely = False


def main(argv):
    print(argv)
    if len(argv) > 0:
        do_train = False
        do_preprocess = False
        do_test = True
        do_build = False
        do_deploy_locally = False
        do_deploy_remotely = False

        if 'deploy-remote' in argv:
            do_deploy_remotely = True
        if 'deploy-local' in argv:
            do_deploy_locally = True
        if 'train' in argv:
            do_train = True
            do_preprocess = True
        if 'build' in argv:
            do_build = True

    cur_dir = os.getcwd()

    if do_train:
        RAW_DATA_DIR = '../../raw-data'
        PROCESSED_DATA_DIR = '../../processed-data'
        os.chdir('train-pipeline')
        # ========== Pre-process NOAA files
        if do_preprocess:
            subprocess.run(['python3', 'preprocessor.py', RAW_DATA_DIR, PROCESSED_DATA_DIR], check=True).check_returncode()

        # ========== (Optional) Select Features
        # TODO

        #  ========== Train Models
        subprocess.run(['python3', 'trainer.py', PROCESSED_DATA_DIR], check=True).check_returncode()

        os.chdir(cur_dir)

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
            subprocess.run(["docker", "stop", "weather-predictor"], check=True).check_returncode()
        except:
            print("Container was not running, OK to proceed")
        try:
            subprocess.run(["docker", "rm", "-f", "weather-predictor"], check=True).check_returncode()
        except:
            print("Container did not exist, OK to proceed")

        subprocess.Popen(
            ["docker", "run", "--name", "weather-predictor", "-v$PWD.", "-p", "5000:5000", "weather-predictor:latest"],
            close_fds=True)

    if do_deploy_remotely:
        os.chdir(cur_dir)
        with open("../credentials_etc.json", 'r') as j:
            credentials_etc = json.loads(j.read())

        # TODO - create repository remotely?

        # Login to the ECR image repo
        p1 = subprocess.Popen(["aws", "ecr", "get-login-password", "--region", credentials_etc["AWS_REGION"]],
                              stdout=subprocess.PIPE)
        subprocess.Popen(["docker", "login", "--username", "AWS", "--password-stdin",
                          "{}.dkr.ecr.{}.amazonaws.com".format(credentials_etc["AWS_ACCOUNT_ID"],
                                                               credentials_etc["AWS_REGION"])],
                         stdin=p1.stdout)

        # Tag the Image
        subprocess.run(["docker", "tag", "weather-predictor:latest",
                        "{}.dkr.ecr.{}.amazonaws.com/weather-predictor:latest".format(credentials_etc["AWS_ACCOUNT_ID"],
                                                                                      credentials_etc["AWS_REGION"])],
                       check=True).check_returncode()

        # Push the image
        subprocess.run(["docker", "push",
                        "{}.dkr.ecr.{}.amazonaws.com/weather-predictor:latest".format(credentials_etc["AWS_ACCOUNT_ID"],
                                                                                      credentials_etc["AWS_REGION"])],
                       check=True).check_returncode()



if __name__ == "__main__":
    main(sys.argv[1:])
