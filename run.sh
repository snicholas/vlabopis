#!/bin/bash
echo "Model start"
pwd
ls -l


if [ -f data/images.tar.gz -a -f data/annotations.tar.gz ]; then
    mkdir -p /tmp/data
    cp data/images.tar.gz /tmp/data/images.tar.gz
    cp data/annotations.tar.gz /tmp/data/annotations.tar.gz
    echo "Unpackaging images.tar.gz"
    tar -xf /tmp/data/images.tar.gz
    echo "Unpackaging annotations.tar.gz"
    tar -xf /tmp/data/annotations.tar.gz
    ls -l data/    
    echo "run training"
    python main.py 1
    echo "ended training"
fi

if [ -f data/oxford_segmentation.h5 -a -f data/images/image.jpg ]; then
    echo "run execution"
    python main.py 0
    echo "ended execution"
    cp data/oxford_segmentation.h5 data/outputs/oxford_segmentation.h5
fi
echo "Model execution ended"
