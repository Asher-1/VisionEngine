#!/bin/bash

mkdir -p build-ubuntu-1804
# shellcheck disable=SC2164
cd build-ubuntu-1804

NCNN_VULKAN=ON
# shellcheck disable=SC2034
WITH_FULL_OPENCV=ON
# shellcheck disable=SC2034
INSTALL_EXAMPLES=ON

cmake -DCMAKE_BUILD_TYPE="Release" \
  -DMIRROR_INSTALL_EXAMPLES=INSTALL_EXAMPLES \
  -DMIRROR_BUILD_WITH_FULL_OPENCV=WITH_FULL_OPENCV \
  -DNCNN_VULKAN=$NCNN_VULKAN ../../..

# shellcheck disable=SC2046
make -j$(nproc)
make install