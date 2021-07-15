#!/bin/bash

INSTALL_DIR=$(pwd)/visionEngine-ios-vulkan
mkdir -p $INSTALL_DIR

BUILD_DIR=build-ios
NCNN_VULKAN=ON
WITH_FULL_OPENCV=OFF
INSTALL_EXAMPLES=OFF
TOOLCHAIN_PATH="../toolchains"
OPENMP_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a"
VULKAN_PATH="/Users/huang/Desktop/ncnn-20210525-full-source/vulkansdk-macos-1.2.162.0"
NCNN_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/ncnn-20210525-ios-vulkan"
OPENCV_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/opencv-mobile-4.5.1-ios"

echo 'Build ios!'
sh compile_ios.sh $TOOLCHAIN_PATH $VULKAN_PATH $NCNN_PATH $OPENCV_PATH $NCNN_VULKAN $WITH_FULL_OPENCV $INSTALL_EXAMPLES $OPENMP_LIBRARY $INSTALL_DIR
rm -rf $BUILD_DIR