FROM python:3.8

MAINTAINER Sergey Numerov <sergey.vn@hotmail.com>

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# For Chrome Driver
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

COPY ./requirements.txt /requirements.txt

WORKDIR /

COPY ./config.py /config.py
COPY ./data-service /data-service
COPY ./webapp /webapp
COPY ./libcommons /libcommons
COPY ./pretrained /pretrained
RUN mkdir -p /logs

COPY ./run.sh /run.sh

RUN pip3 install -r requirements.txt

# This launches both webapps and data-service
ENV FLASK_APP=__init__.py
RUN chmod a+x run.sh
CMD ["./run.sh"]
