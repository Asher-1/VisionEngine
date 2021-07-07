project(visionEngineNcnn)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -std=c++11 -fopenmp")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

set(VISION_ENGINE_DIR ${CMAKE_SOURCE_DIR}/visionEngine-android-vulkan)
include_directories(${VISION_ENGINE_DIR}/${ANDROID_ABI}/include)
set(VISION_ENGINE_LIB ${VISION_ENGINE_DIR}/${ANDROID_ABI}/lib/libVisionEngine.a)
message("VISION_ENGINE_LIB: " ${VISION_ENGINE_LIB})

add_library(visionEngineNcnn SHARED VisionEngineNcnn.cpp ndkcamera.cpp)

target_link_libraries(visionEngineNcnn ncnn ${OpenCV_LIBS} ${VISION_ENGINE_LIB} camera2ndk mediandk)