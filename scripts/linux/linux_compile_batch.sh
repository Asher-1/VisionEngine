#!/bin/bash

INSTALL_DIR=$(pwd)/visionEngine-ubuntu-1804
echo "INSTALLDIR: $INSTALL_DIR"
mkdir -p $INSTALL_DIR

BUILD_DIR=build-ubuntu-1804
NCNN_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/ncnn-20210525-ubuntu-1804"
OPENCV_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/opencv-mobile-4.5.1-ubuntu-1804"

echo 'Build ubuntu 1804!'
sh compile_ubuntu_1804.sh $INSTALL_DIR $NCNN_PATH $OPENCV_PATH
rm -rf $BUILD_DIR