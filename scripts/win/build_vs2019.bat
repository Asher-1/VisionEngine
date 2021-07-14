mkdir vs2019
cd vs2019

NCNN_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/ncnn-20210525-android-vulkan"
OPENCV_PATH="/media/yons/data/develop/git/ncnn/ncnn-android-visionEngine/app/src/main/cpp/opencv-mobile-4.5.1-android"
ANDROID_NDK="/media/yons/data/develop/Android/Sdk/ndk/21.4.7075529"

cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_STL=c++_static -DANDROID_CPP_FEATURES="rtti exceptions" \
    -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-27 \
    -DMIRROR_BUILD_ANDROID=ON -DNCNN_PATH=$NCNN_PATH -DOPENCV_PATH=$OPENCV_PATH \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DCMAKE_BUILD_TYPE="Release" \
    -DMIRROR_BUILD_WITH_FULL_OPENCV=OFF -DNCNN_VULKAN=$NCNN_VULKAN ../../..
