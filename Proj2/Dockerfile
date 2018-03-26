FROM ubuntu:16.04

RUN apt-get update && apt-get upgrade -y

RUN  apt-get install -y \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-openmpi-dev \
    libjpeg-dev \
    libpng-dev

RUN mkdir /code
WORKDIR /code
ADD . .
