cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

message(${SRC_FILES})

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -std=c++11 -fopenmp")
add_library(VisionEngine STATIC ${SRC_FILES})

if (MIRROR_BUILD_ANDROID)
    set(NCNN_PATH "" CACHE PATH "Ncnn root directory")
    set(OPENCV_PATH "" CACHE PATH "Opencv root directory")
else ()
    set(NCNN_PATH "${CMAKE_SOURCE_DIR}/lib/ncnn-20210525-ubuntu-1804" CACHE PATH "Ncnn root directory")
    set(OPENCV_PATH "${CMAKE_SOURCE_DIR}/lib/opencv-mobile-4.5.1-ubuntu-1804" CACHE PATH "Opencv root directory")
endif ()

if (MSVC)
    message("windows")
    include_directories(${CMAKE_SOURCE_DIR}/include/win)
    link_directories(${CMAKE_SOURCE_DIR}/lib/win)
    target_link_libraries(VisionEngine PUBLIC opencv_world344 ncnn)
else ()
    if (MIRROR_BUILD_WITH_FULL_OPENCV)
        find_package(OpenCV REQUIRED)
        target_compile_definitions(VisionEngine PUBLIC MIRROR_BUILD_WITH_FULL_OPENCV)
    else ()
        if (MIRROR_BUILD_ANDROID)
            set(OpenCV_DIR ${OPENCV_PATH}/sdk/native/jni/abi-${ANDROID_ABI})
        else ()
            set(OpenCV_DIR ${OPENCV_PATH}/lib/cmake/opencv4)
        endif ()
        find_package(OpenCV REQUIRED core imgproc highgui)
    endif ()
    message("Opencv Version: ${OPENCV_VERSION}")

    if (MIRROR_BUILD_ANDROID)
        include_directories(${NCNN_PATH}/${ANDROID_ABI}/include)
        set(ncnn_DIR ${NCNN_PATH}/${ANDROID_ABI}/lib/cmake/ncnn)
    else ()
        include_directories(${NCNN_PATH}/include)
        set(ncnn_DIR ${NCNN_PATH}/lib/cmake/ncnn)
    endif ()

    find_package(ncnn REQUIRED)
    target_link_libraries(VisionEngine PUBLIC ncnn ${OpenCV_LIBS})
endif ()

if (MIRROR_OPENMP)
    find_package(OpenMP)
    if (NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        target_compile_options(VisionEngine PRIVATE ${OpenMP_CXX_FLAGS})
    endif ()
endif ()

if (MIRROR_OPENMP AND OpenMP_CXX_FOUND)
    message("Building with OpenMP")
    target_link_libraries(VisionEngine PUBLIC OpenMP::OpenMP_CXX)
endif ()

target_include_directories(VisionEngine
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include/vision>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utility>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/align>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/database>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/living>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/living/live>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detector>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detector/scrfd>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detector/mtcnn>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detector/anticov>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detector/centerface>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detector/retinaface>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/insightface>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/zqlandmarker>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/tracker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/mobilefacenet>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/detectors>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/detectors/yolov4>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/detectors/yolov5>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/detectors/mobilenetssd>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pose>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pose/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pose/detectors/simplepose>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pose/detectors/lightopenpose>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/segment>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/segment/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/segment/detectors/yolact>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/segment/detectors/mobilenetv3>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifiers>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifiers/mobilenet>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifiers/squeezenet>
        )

if (MIRROR_INSTALL_SDK)
    # install head files
    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/common/common.h
            ${CMAKE_CURRENT_SOURCE_DIR}/utility/VisionTools.h
            ${CMAKE_CURRENT_SOURCE_DIR}/face/FaceEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/object/ObjectEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/pose/PoseEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/segment/SegmentEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/classifier/ClassifierEngine.h
            DESTINATION ${PROJECT_BINARY_DIR}/include
            )

    # install library
    install(TARGETS VisionEngine EXPORT VisionEngine ARCHIVE DESTINATION ${PROJECT_BINARY_DIR}/lib)

    if (MIRROR_INSTALL_EXAMPLES)
        # install binaries
        if (MSVC)
            install(FILES
                    ${PROJECT_BINARY_DIR}/src/face.exe
                    ${PROJECT_BINARY_DIR}/src/object.exe
                    ${PROJECT_BINARY_DIR}/src/classifier.exe
                    DESTINATION ${PROJECT_BINARY_DIR}/bin
                    )
        else ()
            install(FILES
                    ${PROJECT_BINARY_DIR}/src/face
                    ${PROJECT_BINARY_DIR}/src/object
                    ${PROJECT_BINARY_DIR}/src/classifier
                    DESTINATION ${PROJECT_BINARY_DIR}/bin
                    )
        endif ()
    endif ()


endif ()

if (MIRROR_BUILD_EXAMPLES)
    # face
    add_executable(face ${CMAKE_SOURCE_DIR}/examples/test_face.cpp)
    target_link_libraries(face PUBLIC VisionEngine)

    # classification
    add_executable(classifier ${CMAKE_SOURCE_DIR}/examples/test_classifier.cpp)
    target_link_libraries(classifier PUBLIC VisionEngine)

    # object
    add_executable(object ${CMAKE_SOURCE_DIR}/examples/test_object.cpp)
    target_link_libraries(object PUBLIC VisionEngine)

    # pose
    add_executable(pose ${CMAKE_SOURCE_DIR}/examples/test_pose.cpp)
    target_link_libraries(pose PUBLIC VisionEngine)

    # segment
    add_executable(segment ${CMAKE_SOURCE_DIR}/examples/test_segment.cpp)
    target_link_libraries(segment PUBLIC VisionEngine)
endif ()