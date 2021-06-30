#!/bin/bash

INSTALL_DIR=visionEngine-ubuntu-1804
BUILD_DIR=build-ubuntu-1804
NCNN_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/ncnn-20210525-ubuntu-1804"
OPENCV_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/opencv-mobile-4.5.1-ubuntu-1804"

echo 'Build ubuntu 1804!'
sh compile_ubuntu_1804.sh $NCNN_PATH $OPENCV_PATH
mkdir -p $INSTALL_DIR
cp -r $BUILD_DIR/include $INSTALL_DIR
cp -r $BUILD_DIR/lib $INSTALL_DIR
cp -r $BUILD_DIR/bin $INSTALL_DIR
rm -rf $BUILD_DIR