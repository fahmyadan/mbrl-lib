#!/bin/bash

eval "$(conda shell.bash hook)"

conda create -n mbrl python=3.10 -y

conda activate mbrl 

pip install -e ".[dev]"

cd mbrl/env/HighwayEnv || exit

pip install -e . 

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia