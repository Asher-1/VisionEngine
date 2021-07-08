#!/bin/bash

INSTALL_DIR=visionEngine-macos-vulkan
BUILD_DIR=build-macos
NCNN_VULKAN=ON
WITH_FULL_OPENCV=OFF
INSTALL_EXAMPLES=OFF
VULKAN_PATH="/Users/huang/Desktop/ncnn-20210525-full-source/vulkansdk-macos-1.2.162.0"
NCNN_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/ncnn-20210525-macos-vulkan"
OPENCV_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/opencv-mobile-4.5.1-macos"

echo 'Build macos!'
sh compile_macos.sh $VULKAN_PATH $NCNN_PATH $OPENCV_PATH $NCNN_VULKAN $WITH_FULL_OPENCV $INSTALL_EXAMPLES
mkdir -p $INSTALL_DIR
cp -r $BUILD_DIR/include $INSTALL_DIR
cp -r $BUILD_DIR/lib $INSTALL_DIR
cp -r $BUILD_DIR/bin $INSTALL_DIR
rm -rf $BUILD_DIR