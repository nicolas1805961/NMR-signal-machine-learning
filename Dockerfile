ARG TAG=py3

FROM python:3.8-slim-buster

WORKDIR /app
# RUN mkdir /app/Repo
# RUN mkdir /app/Repo/ml-python-epita

RUN apt update && apt install -y python3-setuptools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip
RUN apt-get install -y git

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN rm requirements.txt
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/app --ip 0.0.0.0 --no-browser --allow-root"]
