#!/bin/bash

INSTALL_DIR=visionEngine-android-vulkan

echo 'Build android armv7!'
sh compile_android_armv7.sh
mkdir -p $INSTALL_DIR/armeabi-v7a
cp -r build-android-armv7/include $INSTALL_DIR/armeabi-v7a
cp -r build-android-armv7/lib $INSTALL_DIR/armeabi-v7a
rm -rf build-android-armv7

echo 'Build android armv8a!'
sh compile_android_armv8a.sh
mkdir -p $INSTALL_DIR/arm64-v8a
cp -r build-android-armv8a/include $INSTALL_DIR/arm64-v8a
cp -r build-android-armv8a/lib $INSTALL_DIR/arm64-v8a
rm -rf build-android-armv8a

echo 'Build android x86!'
sh compile_android_x86.sh
mkdir -p $INSTALL_DIR/x86
cp -r build-android-x86/include $INSTALL_DIR/x86
cp -r build-android-x86/lib $INSTALL_DIR/x86
rm -rf build-android-x86

echo 'Build android x86_64!'
sh compile_android_x86_64.sh
mkdir -p $INSTALL_DIR/x86_64
cp -r build-android-x86_64/include $INSTALL_DIR/x86_64
cp -r build-android-x86_64/lib $INSTALL_DIR/x86_64
rm -rf build-android-x86_64
