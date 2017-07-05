#!/usr/bin/env bash
# install.sh
# run as root
# workon anom
apt install virtualenv
apt install python-dev python3.5-dev python3-virtualenv python-virtualenv python-pip build-essential gfortran
apt install libpng12-dev libfreetype6-dev
apt install libopenblas-dev liblapack-dev python-scipy python-matplotlib python-numpy
apt install libssl-dev openssl
apt install python-igraph
apt install python-dev python3-dev python3.5-dev python3-virtualenv python-pip build-essential gfortran
apt install canberra-gtk-module
apt install libcanberra-gtk-module*
apt install firefox

mkdir -p ~/src
cd ~/src

git clone git@github.com:totalgood/pugnlp
git clone git@github.com:totalgood/nlpia
git clone git@github.com:totalgood/anomalous

pip install -e ~/src/pugnlp/
pip install -e ~/src/nlpia/
pip install -e ~/src/anomalous/