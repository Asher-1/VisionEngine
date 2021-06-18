cmake_minimum_required(VERSION 3.10)
project(VisionEngine)

Option(MIRROR_OPENMP "openmp support" ON)
Option(MIRROR_BUILD_FACE "build face test" ON)
Option(MIRROR_BUILD_OBJECT "build object test" ON)
Option(MIRROR_BUILD_CLASSIFIER "build classifier test" ON)
Option(MIRROR_BUILD_WITH_FULL_OPENCV "full opencv support" ON)
Option(MIRROR_INSTALL_SDK "install mirror library and headers" ON)

add_subdirectory(src)
