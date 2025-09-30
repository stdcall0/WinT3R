#!/bin/bash

pip install -r requirements.txt
mkdir -p checkpoints
wget -O checkpoints/pytorch_model.bin https://hf-mirror.com/lizizun/WinT3R/resolve/main/pytorch_model.bin
