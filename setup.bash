#!/bin/bash
set -eux

sudo apt-get install -y build-essential bison flex
sudo apt-get install -y --no-install-recommends tk-dev libcairo2-dev

# Sketch
sudo mkdir -p /usr/share/man/man1
sudo apt-get install -y  software-properties-common && \
    sudo add-apt-repository ppa:webupd8team/java -y  && \
    sudo apt-get update && \
    sudo echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    sudo apt-get install -y --allow-unauthenticated oracle-java8-installer && \
    sudo apt-get clean

# conda is interfering during the sketch install
# install sketch by going to sketch-1.7.6/sketch-backend
# chmod +x configure && CC=/usr/bin/gcc ./configure && make -j4 && \
cd sketch-1.7.6/sketch-backend && \
    cd ../sketch-frontend && \
    chmod +x ./sketch && ./sketch test/sk/seq/miniTest1.sk
