#!/bin/bash

INSTALL_DIR=visionEngine-ios-vulkan
BUILD_DIR=build-ios
NCNN_VULKAN=ON
WITH_FULL_OPENCV=OFF
INSTALL_EXAMPLES=OFF
NCNN_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/ncnn-20210525-ios-vulkan"
OPENCV_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/opencv-mobile-4.5.1-ios"

echo 'Build ios!'
sh compile_ios.sh $NCNN_PATH $OPENCV_PATH $NCNN_VULKAN $WITH_FULL_OPENCV $INSTALL_EXAMPLES
mkdir -p $INSTALL_DIR
cp -r $BUILD_DIR/include $INSTALL_DIR
cp -r $BUILD_DIR/lib $INSTALL_DIR
cp -r $BUILD_DIR/bin $INSTALL_DIR
rm -rf $BUILD_DIR