#!/bin/bash

ROOT_DIR=$(pwd)/visionEngine-ios-vulkan
INSTALL_DIR=$ROOT_DIR/visionengine.framework/Versions/A
mkdir -p $INSTALL_DIR/Headers
mkdir -p $INSTALL_DIR/Resources
ln -s A $ROOT_DIR/visionengine.framework/Versions/Current
ln -s Versions/Current/Headers $ROOT_DIR/visionengine.framework/Headers
ln -s Versions/Current/Resources $ROOT_DIR/visionengine.framework/Resources
ln -s Versions/Current/visionengine $ROOT_DIR/visionengine.framework/visionengine

BUILD_DIR=build-ios
NCNN_VULKAN=ON
WITH_FULL_OPENCV=OFF
INSTALL_EXAMPLES=OFF
TOOLCHAIN_PATH="$(pwd)/toolchains"
VULKAN_PATH="/Users/huang/Desktop/ncnn-20210525-full-source/vulkansdk-macos-1.2.162.0"
NCNN_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/ncnn-20210525-ios-vulkan"
OPENCV_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/opencv-mobile-4.5.1-ios"

echo 'Build ios!'
sh compile_ios.sh $TOOLCHAIN_PATH $VULKAN_PATH $NCNN_PATH $OPENCV_PATH $NCNN_VULKAN $WITH_FULL_OPENCV $INSTALL_EXAMPLES $BUILD_DIR $INSTALL_DIR
rm -rf $BUILD_DIR