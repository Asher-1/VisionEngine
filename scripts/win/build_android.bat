@echo off

set ROOT_PATH=%cd%
set INSTALL_DIR=visionEngine-android-vulkan
if not exist %INSTALL_DIR% md %INSTALL_DIR%

set GENERATOR="Ninja"
set ANDROID_PLATFORM="android-28"
set NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-android-vulkan"
set OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-android"
set ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"

:: -------------------armeabi-v7a--------------------
set ANDROID_ABI=armeabi-v7a
echo "Build android %ANDROID_ABI%!"
set BUILD_DIR="build-android-%ANDROID_ABI%"
if not exist %BUILD_DIR% md %BUILD_DIR%
cd %BUILD_DIR%
call:func
cd ..
rmdir %BUILD_DIR% /s /q

:: -------------------arm64-v8a--------------------
set ANDROID_ABI=arm64-v8a
echo "Build android %ANDROID_ABI%!"
set BUILD_DIR="build-android-%ANDROID_ABI%"
if not exist %BUILD_DIR% md %BUILD_DIR%
cd %BUILD_DIR%
call:func
cd ..
rmdir %BUILD_DIR% /s /q

:: -------------------x86--------------------
set ANDROID_ABI=x86
echo "Start build android %ANDROID_ABI%!"
set BUILD_DIR="build-android-%ANDROID_ABI%"
if not exist %BUILD_DIR% md %BUILD_DIR%
cd %BUILD_DIR%
call:func
cd ..
rmdir %BUILD_DIR% /s /q

:: -------------------x86_64--------------------
set ANDROID_ABI=x86_64
echo "Build android %ANDROID_ABI%!"
set BUILD_DIR="build-android-%ANDROID_ABI%"
if not exist %BUILD_DIR% md %BUILD_DIR%
cd %BUILD_DIR%
call:func
cd ..
rmdir %BUILD_DIR% /s /q

:: -----------------------------------------------cmake function-------------------------------------------------
:func
echo "Temp install dir: %ROOT_PATH%/%INSTALL_DIR%/%ANDROID_ABI%"
cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CONFIGURATION_TYPES="Release" ^
      -DCMAKE_INSTALL_PREFIX="%ROOT_PATH%/%INSTALL_DIR%/%ANDROID_ABI%" ^
      -DCMAKE_GENERATOR=%GENERATOR% ^
      -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" ^
      -DANDROID_NDK=%ANDROID_NDK% ^
      -DANDROID_ABI=%ANDROID_ABI% ^
      -DANDROID_PLATFORM=%ANDROID_PLATFORM% ^
      -DMIRROR_BUILD_ANDROID=ON ^
      -DNCNN_PATH=%NCNN_PATH% ^
      -DOPENCV_PATH=%OPENCV_PATH% ^
      -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF ^
      -DNCNN_VULKAN=ON ../../..
Ninja
Ninja install
echo "Build android %ANDROID_ABI% successfully!"
goto:eof