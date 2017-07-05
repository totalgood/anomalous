# Installation on Ubuntu

Run these commands as root

```shell
# sudo -i
apt install -y python-dev python3.5-dev python3-virtualenv python-virtualenv python-pip build-essential gfortran
apt install -y libpng12-dev libfreetype6-dev
apt install -y libopenblas-dev liblapack-dev python-scipy python-matplotlib python-numpy
apt install -y libssl-dev openssl
apt install -y python-igraph
apt install -y python3-dev python3-pip python3.5-dev python3-virtualenv python-pip build-essential gfortran
apt install -y canberra-gtk-module
apt install -y libcanberra-gtk-module*
apt install -y firefox

mkdir -p ~/src
cd ~/src

git clone git@github.com:totalgood/pugnlp
git clone git@github.com:totalgood/nlpia
git clone git@github.com:totalgood/anomalous

pip3 install -e ~/src/pugnlp/
pip3 install -e ~/src/nlpia/
pip3 install -e ~/src/anomalous/

cp ~/src/anomalous/secrets.cfg.EXAMPLE_TEMPLATE ~/src/anomalous/secrets.cfg
```