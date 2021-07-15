mkdir build-android
cd build-android

%ANDROID_SDK_HOME%/cmake/3.10.2.4988404/bin/cmake ^
  -DANDROID_ABI=%abi% ^
  -DANDROID_NDK=%ANDROID_SDK_HOME%/ndk-bundle ^

set ANDROID_PLATFORM="android-27"
set ANDROID_ABI="armeabi-v7a"
set NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-android-vulkan"
set OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-android"
set ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" ^
      -DCMAKE_INSTALL_PREFIX=%cd%/visionEngine-android-vulkan ^
      -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" ^
      -DCMAKE_MAKE_PROGRAME="%ANDROID_NDK%/prebuilt/windows/bin/make.exe" -DANDROID_PLATFORM=%ANDROID_PLATFORM% ^
      -DMIRROR_BUILD_ANDROID=ON -DNCNN_PATH=%NCNN_PATH% -DOPENCV_PATH=%OPENCV_PATH% -DANDROID_ABI=%ANDROID_ABI% ^
      -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF -DNCNN_VULKAN=ON ../../..
nmake
nmake install