cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
message(${SRC_FILES})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -std=c++11")

add_library(VisionEngine STATIC ${SRC_FILES})

if (MIRROR_BUILD_ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
    set(NCNN_PATH "" CACHE PATH "Ncnn root directory")
    set(OPENCV_PATH "" CACHE PATH "Opencv root directory")
elseif (MIRROR_BUILD_IOS)
    set(VULKAN_PATH "" CACHE PATH "vulkan root directory")
    set(NCNN_PATH "" CACHE PATH "Ncnn root directory")
    set(OPENCV_PATH "" CACHE PATH "Opencv root directory")
else ()
    set(NCNN_PATH "${CMAKE_SOURCE_DIR}/lib/ncnn-20210525-ubuntu-1804" CACHE PATH "Ncnn root directory")
    set(OPENCV_PATH "${CMAKE_SOURCE_DIR}/lib/opencv-mobile-4.5.1-ubuntu-1804" CACHE PATH "Opencv root directory")
endif ()

message("NCNN_PATH: ${NCNN_PATH}")
message("OPENCV_PATH: ${OPENCV_PATH}")

if (MIRROR_BUILD_WITH_FULL_OPENCV)
	find_package(OpenCV REQUIRED)
	target_compile_definitions(VisionEngine PUBLIC MIRROR_BUILD_WITH_FULL_OPENCV)
else ()
	if (WIN32 AND NOT MIRROR_BUILD_ANDROID)
		set(OpenCV_DIR ${OPENCV_PATH}/x64/x64/vc16/staticlib)
		find_package(OpenCV REQUIRED core imgproc highgui)
	elseif (MIRROR_BUILD_ANDROID)
		set(OpenCV_DIR ${OPENCV_PATH}/sdk/native/jni/abi-${ANDROID_ABI})
		find_package(OpenCV REQUIRED core imgproc highgui)
	elseif (MIRROR_BUILD_IOS)
		link_directories(${OPENCV_PATH})
		include_directories(SYSTEM ${OPENCV_PATH})
		include_directories(${OPENCV_PATH}/opencv2.framework)
	else ()
		set(OpenCV_DIR ${OPENCV_PATH}/lib/cmake/opencv4)
		find_package(OpenCV REQUIRED core imgproc highgui)
	endif ()
endif ()
message("Opencv Version: ${OPENCV_VERSION}")

if (WIN32 AND NOT MIRROR_BUILD_ANDROID)
	add_definitions(-D_USE_MATH_DEFINES)
	include_directories(${NCNN_PATH}/x64/include)
	set(ncnn_DIR ${NCNN_PATH}/x64/lib/cmake/ncnn)
	find_package(ncnn REQUIRED)
	target_link_libraries(VisionEngine PUBLIC ncnn ${OpenCV_LIBS})
else ()
    if (MIRROR_BUILD_ANDROID)
        include_directories(${NCNN_PATH}/${ANDROID_ABI}/include)
        set(ncnn_DIR ${NCNN_PATH}/${ANDROID_ABI}/lib/cmake/ncnn)
    elseif (MIRROR_BUILD_IOS)
        link_directories(${NCNN_PATH})
        include_directories(SYSTEM ${NCNN_PATH})
        include_directories(${VULKAN_PATH}/MoltenVK/include)
        include_directories(${NCNN_PATH}/glslang.framework)
        include_directories(${NCNN_PATH}/openmp.framework)
        include_directories(${NCNN_PATH}/ncnn.framework/Headers)
    else ()
        include_directories(${NCNN_PATH}/include)
        set(ncnn_DIR ${NCNN_PATH}/lib/cmake/ncnn)
    endif ()

    if (MIRROR_BUILD_IOS)
        set(CMAKE_FRAMEWORK_PATH "${CMAKE_FRAMEWORK_PATH} ${NCNN_PATH} ${OPENCV_PATH}")
        set(CMAKE_SYSTEM_FRAMEWORK_PATH "${CMAKE_SYSTEM_FRAMEWORK_PATH} ${NCNN_PATH} ${OPENCV_PATH}")
        message("CMAKE_SYSTEM_FRAMEWORK_PATH: " ${CMAKE_SYSTEM_FRAMEWORK_PATH})
        set_target_properties(VisionEngine PROPERTIES
                LINK_FLAGS "-Wl,-F${NCNN_PATH}/ncnn.framework -F${NCNN_PATH}/openmp.framework -F${NCNN_PATH}/glslang.framework"
                FRAMEWORK_SEARCH_PATHS ${NCNN_PATH})
        set_target_properties(VisionEngine PROPERTIES
                LINK_FLAGS "-Wl,-F${OPENCV_PATH}/opencv2.framework"
                FRAMEWORK_SEARCH_PATHS ${OPENCV_PATH})
    else ()
        find_package(ncnn REQUIRED)
        target_link_libraries(VisionEngine PUBLIC ncnn ${OpenCV_LIBS})
    endif ()
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
    if (NOT MIRROR_BUILD_IOS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/detectors/nanodet>
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

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr/utils>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr/detectors>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr/detectors/dbnet>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr/recognizers>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ocr/recognizers/crnn>
        )

if (MIRROR_INSTALL_SDK)
    # install head files
    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/common/common.h
            ${CMAKE_CURRENT_SOURCE_DIR}/utility/VisionTools.h
            ${CMAKE_CURRENT_SOURCE_DIR}/ocr/OcrEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/face/FaceEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/object/ObjectEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/pose/PoseEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/segment/SegmentEngine.h
            ${CMAKE_CURRENT_SOURCE_DIR}/classifier/ClassifierEngine.h
            DESTINATION include
            )

    # install library
    install(TARGETS VisionEngine EXPORT VisionEngine ARCHIVE DESTINATION lib)

    if (MIRROR_INSTALL_EXAMPLES)
        # install binaries
        if (UNIX)
            install(FILES
                    ${PROJECT_BINARY_DIR}/src/ocr
                    ${PROJECT_BINARY_DIR}/src/pose
                    ${PROJECT_BINARY_DIR}/src/segment
                    ${PROJECT_BINARY_DIR}/src/face
                    ${PROJECT_BINARY_DIR}/src/object
                    ${PROJECT_BINARY_DIR}/src/classifier
                    DESTINATION bin
                    )
        elseif (WIN32 AND NOT MIRROR_BUILD_ANDROID)
            install(FILES
                    ${PROJECT_BINARY_DIR}/src/$<CONFIG>/ocr.exe
                    ${PROJECT_BINARY_DIR}/src/$<CONFIG>/pose.exe
                    ${PROJECT_BINARY_DIR}/src/$<CONFIG>/segment.exe
                    ${PROJECT_BINARY_DIR}/src/$<CONFIG>/face.exe
                    ${PROJECT_BINARY_DIR}/src/$<CONFIG>/object.exe
                    ${PROJECT_BINARY_DIR}/src/$<CONFIG>/classifier.exe
                    DESTINATION bin
                    )
        else ()
            #set_target_properties(VisionEngine PROPERTIES FRAMEWORK TRUE)
            install(FILES
                    ${PROJECT_BINARY_DIR}/src/ocr
                    ${PROJECT_BINARY_DIR}/src/pose
                    ${PROJECT_BINARY_DIR}/src/segment
                    ${PROJECT_BINARY_DIR}/src/face
                    ${PROJECT_BINARY_DIR}/src/object
                    ${PROJECT_BINARY_DIR}/src/classifier
                    DESTINATION bin
                    )
        endif ()
    endif ()


endif ()

if (MIRROR_BUILD_EXAMPLES)
    if (APPLE)
        # face
        add_executable(face ${CMAKE_SOURCE_DIR}/examples/test_face.mm)
        target_link_libraries(face PRIVATE VisionEngine)

        # classification
        add_executable(classifier ${CMAKE_SOURCE_DIR}/examples/test_classifier.mm)
        target_link_libraries(classifier PRIVATE VisionEngine)

        # object
        add_executable(object ${CMAKE_SOURCE_DIR}/examples/test_object.mm)
        target_link_libraries(object PRIVATE VisionEngine)

        # pose
        add_executable(pose ${CMAKE_SOURCE_DIR}/examples/test_pose.mm)
        target_link_libraries(pose PRIVATE VisionEngine)

        # segment
        add_executable(segment ${CMAKE_SOURCE_DIR}/examples/test_segment.mm)
        target_link_libraries(segment PRIVATE VisionEngine)

        # ocr
        add_executable(ocr ${CMAKE_SOURCE_DIR}/examples/test_ocr.mm)
        target_link_libraries(ocr PRIVATE VisionEngine)
    else ()
        # face
        add_executable(face ${CMAKE_SOURCE_DIR}/examples/test_face.cpp)
        target_link_libraries(face PRIVATE VisionEngine)

        # classification
        add_executable(classifier ${CMAKE_SOURCE_DIR}/examples/test_classifier.cpp)
        target_link_libraries(classifier PRIVATE VisionEngine)

        # object
        add_executable(object ${CMAKE_SOURCE_DIR}/examples/test_object.cpp)
        target_link_libraries(object PRIVATE VisionEngine)

        # pose
        add_executable(pose ${CMAKE_SOURCE_DIR}/examples/test_pose.cpp)
        target_link_libraries(pose PRIVATE VisionEngine)

        # segment
        add_executable(segment ${CMAKE_SOURCE_DIR}/examples/test_segment.cpp)
        target_link_libraries(segment PRIVATE VisionEngine)

        # ocr
        add_executable(ocr ${CMAKE_SOURCE_DIR}/examples/test_ocr.cpp)
        target_link_libraries(ocr PRIVATE VisionEngine)
    endif ()

endif ()