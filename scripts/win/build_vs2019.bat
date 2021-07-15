@echo off

if not exist vs2019 md vs2019
cd vs2019

set INSTALL_DIR="visionEngine-windows-vs2019"
if not exist %INSTALL_DIR% md %INSTALL_DIR%

set NCNN_VULKAN=ON
set INSTALL_EXAMPLES=ON
set WITH_FULL_OPENCV=OFF
set NCNN_PATH="E:/ai/projects/VisionEngine/lib/ncnn-20210525-windows-vs2019"
set OPENCV_PATH="E:/ai/projects/VisionEngine/lib/opencv-mobile-4.5.1-windows-vs2019"

cmake -G "Visual Studio 16 2019" -A x64 ^
	  -DCMAKE_BUILD_TYPE="Release" ^
      -DCMAKE_INSTALL_PREFIX=%cd%/%INSTALL_DIR% ^
      -DMIRROR_INSTALL_EXAMPLES=%INSTALL_EXAMPLES% ^
      -DNCNN_PATH=%NCNN_PATH% ^
	  -DOPENCV_PATH=%OPENCV_PATH% ^
      -DMIRROR_BUILD_WITH_FULL_OPENCV=%WITH_FULL_OPENCV% ^
	  -DNCNN_VULKAN=%NCNN_VULKAN% ../../..
pause