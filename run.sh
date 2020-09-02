#!/bin/bash

mkdir -p /data/outputs
if [ -f /data/images.tar.gz -a -f /data/annotations.tar.gz ]; then
    tar -xf /data/images.tar.gz
    tar -xf /data/annotations.tar.gz
    echo "run training"
    python main.py 1
fi

if [ -f /data/oxford_segmentation.h5 -a -f /data/image.png ]; then
    cp /data/oxford_segmentation.h5 /data/outputs/oxford_segmentation.h5
    echo "run execution"
    python main.py 0
fi

