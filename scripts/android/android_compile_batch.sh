#!/bin/bash

INSTALL_DIR=visionEngine-android-vulkan

ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"
NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-android-vulkan"
OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-android"

echo 'Build android armv7!'
sh compile_android_armv7.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK
mkdir -p $INSTALL_DIR/armeabi-v7a
cp -r build-android-armv7/include $INSTALL_DIR/armeabi-v7a
cp -r build-android-armv7/lib $INSTALL_DIR/armeabi-v7a
rm -rf build-android-armv7

echo 'Build android armv8a!'
sh compile_android_armv8a.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK
mkdir -p $INSTALL_DIR/arm64-v8a
cp -r build-android-armv8a/include $INSTALL_DIR/arm64-v8a
cp -r build-android-armv8a/lib $INSTALL_DIR/arm64-v8a
rm -rf build-android-armv8a

echo 'Build android x86!'
sh compile_android_x86.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK
mkdir -p $INSTALL_DIR/x86
cp -r build-android-x86/include $INSTALL_DIR/x86
cp -r build-android-x86/lib $INSTALL_DIR/x86
rm -rf build-android-x86

echo 'Build android x86_64!'
sh compile_android_x86_64.sh $NCNN_PATH $OPENCV_PATH $ANDROID_NDK
mkdir -p $INSTALL_DIR/x86_64
cp -r build-android-x86_64/include $INSTALL_DIR/x86_64
cp -r build-android-x86_64/lib $INSTALL_DIR/x86_64
rm -rf build-android-x86_64
