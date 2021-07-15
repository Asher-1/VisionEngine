@echo off

if not exist vs2019 md vs2019
cd vs2019

set INSTALL_EXAMPLES=ON
set INSTALL_DIR="visionEngine-windows-vs2019"
set NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-windows-vs2019"
set OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-windows-vs2019"

cmake -G "Visual Studio 15 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CONFIGURATION_TYPES="Release" ^
      -DCMAKE_INSTALL_PREFIX=%cd%/%INSTALL_DIR% ^
      -DMIRROR_INSTALL_EXAMPLES=%INSTALL_EXAMPLES% ^
      -DNCNN_PATH=%NCNN_PATH% -DOPENCV_PATH=%OPENCV_PATH% ^
      -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF -DNCNN_VULKAN=ON ../../..
pause