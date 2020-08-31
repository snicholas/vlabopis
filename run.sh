#!/bin/bash
tar -xf /data/images.tar.gz
tar -xf /data/annotations.tar.gz
mkdir -p /data/outputs
pip install -r requirements.txt
python main.py