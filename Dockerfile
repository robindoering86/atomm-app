FROM ubuntu:18.10
LABEL maintainer="Robin C. Doering <robin@atomm.de>"

RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
RUN apt-get update && apt-get install -y git

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install gunicorn

COPY ./ /app
WORKDIR /app
EXPOSE 8001
CMD gunicorn --bind 0.0.0.0:8001 index:server
