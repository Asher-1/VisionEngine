#!/bin/bash

mkdir -p build-android-x86
# shellcheck disable=SC2164
cd build-android-x86

NCNN_VULKAN=ON
if [ $# != 3 ]; then
  NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/jni/ncnn-20210525-android-vulkan"
  OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/jni/opencv-mobile-4.5.1-android"
  ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"
else
  NCNN_PATH="$1"
  OPENCV_PATH="$2"
  ANDROID_NDK="$3"
fi

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"
echo "ANDROID_NDK: $ANDROID_NDK"

# If you want to enable Vulkan, platform api version >= android-24 is needed
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DMIRROR_BUILD_ANDROID=ON -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
  -DANDROID_ABI="x86" -DCMAKE_BUILD_TYPE="Release" \
  -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF -DANDROID_PLATFORM=android-27 -DNCNN_VULKAN=$NCNN_VULKAN ../../..

# shellcheck disable=SC2046
make -j$(nproc)
make install