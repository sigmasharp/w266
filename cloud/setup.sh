#!/bin/bash

set -x

sudo apt-get upgrade
sudo apt-get update

wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
bash Anaconda2-4.1.1-Linux-x86_64.sh

source /home/${USER}/.bashrc
conda install -c jjhelmus tensorflow
jupyter notebook --generate-config
cp support/jupyter_notebook_config.py /home/${USER}/.jupyter/jupyter_notebook_config.py
