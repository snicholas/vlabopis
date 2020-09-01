#!/bin/bash
tar -xf /data/images.tar.gz
tar -xf /data/annotations.tar.gz
mkdir -p /data/outputs
python main.py