#!/bin/bash
echo "Model start"

if [ -f data/images.tar.gz -a -f data/annotations.tar.gz ]; then
    cd data
    ls -lh 
    echo "Unpackaging images.tar.gz"
    tar -xf data/images.tar.gz
    echo "Unpackaging annotations.tar.gz"
    tar -xf data/annotations.tar.gz
    ls -lh
    cd ..
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
