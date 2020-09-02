#!/bin/bash
MODE="execute"
mkdir -p /data/outputs
if [-f /data/images.tar.gz -a -f /data/annotations.tar.gz]; then
    tar -xf /data/images.tar.gz
    tar -xf /data/annotations.tar.gz
    python main.py
fi

if [-f /data/oxford_segmentation.h5 -a -f /data/image.png]; then
    python execute.py
fi

