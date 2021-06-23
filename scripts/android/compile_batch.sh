#!/bin/bash

echo 'Build android armv7!'
sh compile_android_armv7.sh
mkdir -p visionEngine/armeabi-v7a
cp -r build-android-armv7/include visionEngine/armeabi-v7a
cp -r build-android-armv7/lib visionEngine/armeabi-v7a

echo 'Build android armv8a!'
sh compile_android_armv8a.sh
mkdir -p visionEngine/arm64-v8a
cp -r build-android-armv8a/include visionEngine/arm64-v8a
cp -r build-android-armv8a/lib visionEngine/arm64-v8a

echo 'Build android x86!'
sh compile_android_x86.sh
mkdir -p visionEngine/x86
cp -r build-android-x86/include visionEngine/x86
cp -r build-android-x86/lib visionEngine/x86

echo 'Build android x86_64!'
sh compile_android_x86_64.sh
mkdir -p visionEngine/x86_64
cp -r build-android-x86_64/include visionEngine/x86_64
cp -r build-android-x86_64/lib visionEngine/x86_64

