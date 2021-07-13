cmake_minimum_required(VERSION 3.14)
project(VisionEngine)

Option(MIRROR_OPENMP "openmp support" ON)
Option(MIRROR_BUILD_ANDROID "build android" OFF)
Option(MIRROR_BUILD_IOS "build android" OFF)
Option(MIRROR_BUILD_EXAMPLES "build examples" ON)
Option(MIRROR_BUILD_WITH_FULL_OPENCV "full opencv support" ON)
Option(MIRROR_INSTALL_SDK "install mirror library and headers" ON)
Option(MIRROR_INSTALL_EXAMPLES "install mirror examples" OFF)

add_subdirectory(src)
