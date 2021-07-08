#!/bin/bash

mkdir -p build-ios
cd build-ios

if [ $# != 5 ]; then
  NCNN_VULKAN=ON
  WITH_FULL_OPENCV=ON
  INSTALL_EXAMPLES=ON
  NCNN_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/ncnn-20210525-ios-vulkan"
  OPENCV_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/opencv-mobile-4.5.1-ios"
else
  NCNN_PATH="$1"
  OPENCV_PATH="$2"
  NCNN_VULKAN="$3"
  WITH_FULL_OPENCV="$4"
  INSTALL_EXAMPLES="$5"
fi

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"

if [ $NCNN_VULKAN == 'ON'  ]; then
# vulkan is only available on arm64 devices
cmake -DIOS_PLATFORM=OS64 -DIOS_ARCH="arm64;arm64e" \
      -DMIRROR_BUILD_IOS=ON -DCMAKE_BUILD_TYPE="Release" \
      -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
      -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
      -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
      -DNCNN_VULKAN=$NCNN_VULKAN ../../..
else
  cmake -DIOS_PLATFORM=OS -DIOS_ARCH="armv7;arm64;arm64e" \
      -DMIRROR_BUILD_IOS=ON -DCMAKE_BUILD_TYPE="Release" \
      -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
      -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
      -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
      -DNCNN_VULKAN=$NCNN_VULKAN ../../..
fi

cmake --build . -j$(nproc)
cmake --build . --target install