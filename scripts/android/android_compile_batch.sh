#!/bin/bash

INSTALL_DIR=$(pwd)/visionEngine-android-vulkan
mkdir -p "$INSTALL_DIR"

NCNN_VULKAN=ON
ANDROID_PLATFORM="android-28"
ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"
NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-android-vulkan"
OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-android"

ANDROID_ABI=armeabi-v7a
echo "Start build android $ANDROID_ABI!"
BUILD_DIR="build-android-$ANDROID_ABI"
sh compile_android_armv8a.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK $ANDROID_PLATFORM $NCNN_VULKAN $BUILD_DIR $INSTALL_DIR/$ANDROID_ABI
echo "Install dir: $INSTALL_DIR/$ANDROID_ABI"
rm -rf $BUILD_DIR

ANDROID_ABI=arm64-v8a
echo "Start build android $ANDROID_ABI!"
BUILD_DIR="build-android-$ANDROID_ABI"
sh compile_android_armv8a.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK $ANDROID_PLATFORM $NCNN_VULKAN $BUILD_DIR $INSTALL_DIR/$ANDROID_ABI
echo "Install dir: $INSTALL_DIR/$ANDROID_ABI"
rm -rf $BUILD_DIR

ANDROID_ABI=x86
echo "Start build android $ANDROID_ABI!"
BUILD_DIR="build-android-$ANDROID_ABI"
sh compile_android_x86.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK $ANDROID_PLATFORM $NCNN_VULKAN $BUILD_DIR $INSTALL_DIR/$ANDROID_ABI
echo "Install dir: $INSTALL_DIR/$ANDROID_ABI"
rm -rf $BUILD_DIR

ANDROID_ABI=x86_64
echo "Start build android $ANDROID_ABI!"
BUILD_DIR="build-android-$ANDROID_ABI"
sh compile_android_x86_64.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK $ANDROID_PLATFORM $NCNN_VULKAN $BUILD_DIR $INSTALL_DIR/$ANDROID_ABI
echo "Install dir: $INSTALL_DIR/$ANDROID_ABI"
rm -rf $BUILD_DIR
