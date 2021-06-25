#!/bin/bash

mkdir -p build-android-armv7
# shellcheck disable=SC2164
cd build-android-armv7

NCNN_VULKAN=ON
ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"

# If you want to enable Vulkan, platform api version >= android-24 is needed
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DMIRROR_BUILD_ANDROID=ON -DCMAKE_BUILD_TYPE="Release" \
  -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF -DANDROID_PLATFORM=android-27 -DNCNN_VULKAN=$NCNN_VULKAN ../../..

# shellcheck disable=SC2046
make -j$(nproc)
make install