#!/bin/bash

mkdir -p build-macos
cd build-macos

if [ $# != 6 ]; then
  NCNN_VULKAN=ON
  WITH_FULL_OPENCV=OFF
  INSTALL_EXAMPLES=OFF
  VULKAN_PATH="/Users/huang/Desktop/ncnn-20210525-full-source/vulkansdk-macos-1.2.162.0"
  NCNN_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/ncnn-20210525-macos-vulkan"
  OPENCV_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/opencv-mobile-4.5.1-macos"
else
  VULKAN_PATH="$1"
  NCNN_PATH="$2"
  OPENCV_PATH="$3"
  NCNN_VULKAN="$4"
  WITH_FULL_OPENCV="$5"
  INSTALL_EXAMPLES="$6"
fi

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"

# vulkan is only available on arm64 devices
cmake -DMIRROR_BUILD_IOS=OFF -DCMAKE_BUILD_TYPE="Release" \
      -DCMAKE_OSX_ARCHITECTURES="x86_64" \
      -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
      -DMIRROR_BUILD_EXAMPLES=ON -DVULKAN_PATH=$VULKAN_PATH \
      -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
      -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
      -DNCNN_VULKAN=$NCNN_VULKAN ../../..

cmake --build . -j4
cmake --build . --target install