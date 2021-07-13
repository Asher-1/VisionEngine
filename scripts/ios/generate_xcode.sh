#!/bin/bash

mkdir -p build-ios
cd build-ios

if [ $# != 8 ]; then
  NCNN_VULKAN=ON
  WITH_FULL_OPENCV=OFF
  INSTALL_EXAMPLES=OFF
  TOOLCHAIN_PATH="../toolchains"
  VULKAN_PATH="/Users/huang/Desktop/ncnn-20210525-full-source/vulkansdk-macos-1.2.162.0"
  NCNN_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/ncnn-20210525-ios-vulkan"
  OPENCV_PATH="/Users/huang/Desktop/Cpp-VisionEigen/lib/opencv-mobile-4.5.1-ios"
  OPENMP_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a"
else
  TOOLCHAIN_PATH="$1"
  VULKAN_PATH="$2"
  NCNN_PATH="$3"
  OPENCV_PATH="$4"
  NCNN_VULKAN="$5"
  WITH_FULL_OPENCV="$6"
  INSTALL_EXAMPLES="$7"
  OPENMP_LIBRARY="$8"
fi

echo "NCNN_PATH: $NCNN_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"

if [ $NCNN_VULKAN == 'ON'  ]; then
# vulkan is only available on arm64 devices
cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH/ios.toolchain.cmake \
      -DIOS_PLATFORM=OS64 -DIOS_ARCH="arm64;arm64e" \
      -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
      -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
      -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
      -DOpenMP_libomp_LIBRARY=$OPENMP_LIBRARY \
      -DMIRROR_BUILD_IOS=ON -DCMAKE_BUILD_TYPE="Release" \
      -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
      -DMIRROR_BUILD_EXAMPLES=OFF -DVULKAN_PATH=$VULKAN_PATH \
      -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
      -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
      -DNCNN_VULKAN=$NCNN_VULKAN -G "Xcode" ../../..
else
  cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH/ios.toolchain.cmake \
        -DIOS_PLATFORM=OS -DIOS_ARCH="armv7;arm64;arm64e" \
        -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
        -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
        -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
        -DOpenMP_libomp_LIBRARY=$OPENMP_LIBRARY \
        -DMIRROR_BUILD_IOS=ON -DCMAKE_BUILD_TYPE="Release" \
        -DMIRROR_INSTALL_EXAMPLES=$INSTALL_EXAMPLES \
        -DMIRROR_BUILD_EXAMPLES=OFF -DVULKAN_PATH=$VULKAN_PATH \
        -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
        -DMIRROR_BUILD_WITH_FULL_OPENCV=$WITH_FULL_OPENCV \
        -DNCNN_VULKAN=$NCNN_VULKAN -G "Xcode" ../../..
fi

cmake --build . -j$(nproc)
cmake --build . --target install