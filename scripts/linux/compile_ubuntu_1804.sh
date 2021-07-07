#!/bin/bash

mkdir -p build-ubuntu-1804
# shellcheck disable=SC2164
cd build-ubuntu-1804

NCNN_VULKAN=ON
# shellcheck disable=SC2034
WITH_FULL_OPENCV=ON
# shellcheck disable=SC2034
INSTALL_EXAMPLES=ON

if [ $# != 2 ]; then
  NCNN_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/ncnn-20210525-ubuntu-1804"
  OPENCV_PATH="/media/yons/data/develop/git/ncnn/VisionEngine/lib/opencv-mobile-4.5.1-ubuntu-1804"
else
  NCNN_PATH="$1"
  OPENCV_PATH="$2"
fi

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"

cmake -DCMAKE_BUILD_TYPE="Release" \
  -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
  -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
  -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
  -DNCNN_VULKAN=$NCNN_VULKAN ../../..

# shellcheck disable=SC2046
make -j$(nproc)
make install