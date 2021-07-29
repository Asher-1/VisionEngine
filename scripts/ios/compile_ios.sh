#!/bin/bash

if [ $# != 9 ]; then
  NCNN_VULKAN=ON
  WITH_FULL_OPENCV=OFF
  INSTALL_EXAMPLES=OFF
  TOOLCHAIN_PATH="$(pwd)/toolchains"
  VULKAN_PATH="/Users/huang/Desktop/ncnn-20210525-full-source/vulkansdk-macos-1.2.162.0"
  NCNN_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/ncnn-20210525-ios-vulkan"
  OPENCV_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/opencv-mobile-4.5.1-ios"
  BUILD_DIR=build-ios
  ROOT_DIR=$(pwd)/visionEngine-ios-vulkan
  INSTALL_DIR=$ROOT_DIR/visionengine.framework/Versions/A
  mkdir -p $INSTALL_DIR/Headers
  mkdir -p $INSTALL_DIR/Resources
  ln -s A $ROOT_DIR/visionengine.framework/Versions/Current
  ln -s Versions/Current/Headers $ROOT_DIR/visionengine.framework/Headers
  ln -s Versions/Current/Resources $ROOT_DIR/visionengine.framework/Resources
  ln -s Versions/Current/visionengine $ROOT_DIR/visionengine.framework/visionengine
else
  TOOLCHAIN_PATH="$1"
  VULKAN_PATH="$2"
  NCNN_PATH="$3"
  OPENCV_PATH="$4"
  NCNN_VULKAN="$5"
  WITH_FULL_OPENCV="$6"
  INSTALL_EXAMPLES="$7"
  BUILD_DIR="$8"
  INSTALL_DIR="$9"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"

if [ $NCNN_VULKAN == 'ON'  ]; then
# vulkan is only available on arm64 devices
cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH/ios.toolchain.cmake \
      -DCMAKE_BUILD_TYPE="Release" \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DIOS_PLATFORM=OS64  \
      -DIOS_ARCH="arm64;arm64e" \
      -DMIRROR_BUILD_IOS=ON \
      -DMIRROR_BUILD_EXAMPLES=OFF \
      -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
      -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
      -DENABLE_BITCODE=0 \
      -DENABLE_ARC=0 \
      -DENABLE_VISIBILITY=0 \
      -DOPENCV_PATH=$OPENCV_PATH \
      -DNCNN_PATH=$NCNN_PATH \
      -DVULKAN_PATH=$VULKAN_PATH \
      -DVulkan_INCLUDE_DIR=$VULKAN_PATH/MoltenVK/include \
      -DVulkan_LIBRARY=$VULKAN_PATH/MoltenVK/dylib/iOS/libMoltenVK.dylib \
      -DNCNN_VULKAN=$NCNN_VULKAN ../../..
else
  cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH/ios.toolchain.cmake \
        -DCMAKE_BUILD_TYPE="Release" \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DIOS_PLATFORM=OS \
        -DIOS_ARCH="armv7;arm64;arm64e" \
        -DMIRROR_BUILD_IOS=ON \
        -DMIRROR_BUILD_EXAMPLES=OFF \
        -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
        -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
        -DENABLE_BITCODE=0 \
        -DENABLE_ARC=0 \
        -DENABLE_VISIBILITY=0 \
        -DOPENCV_PATH=$OPENCV_PATH \
        -DNCNN_PATH=$NCNN_PATH \
        -DNCNN_VULKAN=$NCNN_VULKAN ../../..
fi

cmake --build . --config Release
cmake --install . --config Release