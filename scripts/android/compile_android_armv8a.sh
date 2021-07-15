#!/bin/bash


NCNN_VULKAN=ON
if [ $# != 7 ]; then
  NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-android-vulkan"
  OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-android"
  ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"
  ANDROID_PLATFORM="android-28"
  NCNN_VULKAN=ON
  BUILD_DIR=build-android-armv8a
  INSTALL_DIR=$(pwd)/$BUILD_DIR
else
  NCNN_PATH="$1"
  OPENCV_PATH="$2"
  ANDROID_NDK="$3"
  ANDROID_PLATFORM="$4"
  NCNN_VULKAN="$5"
  BUILD_DIR="$6"
  INSTALL_DIR="$7"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"
echo "ANDROID_NDK: $ANDROID_NDK"

# If you want to enable Vulkan, platform api version >= android-24 is needed
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DANDROID_STL=c++_static -DANDROID_CPP_FEATURES="rtti exceptions" \
  -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=$ANDROID_PLATFORM \
  -DMIRROR_BUILD_ANDROID=ON -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
  -DANDROID_ABI="arm64-v8a" -DCMAKE_BUILD_TYPE="Release" \
  -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF -DNCNN_VULKAN=$NCNN_VULKAN ../../..

# shellcheck disable=SC2046
make -j$(nproc)
make install