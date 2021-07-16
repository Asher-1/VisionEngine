# VisionEngine

## 预先准备

Visual Studio 2017 Community Edition，使用动态的 CRT 运行库

以下命令行均使用  **适用于 VS 2017 的 x64 本机工具命令提示**

## 编译安装 protobuf

https://github.com/google/protobuf/archive/v3.4.0.zip

 [windows-x64-cpu-vs2017.yml](../../.github/workflows/windows-x64-cpu-vs2017.yml) 我下载到 C:/Users/shuiz/source 解压缩

```batch
mkdir build-vs2017
cd build-vs2017
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install ^
    -Dprotobuf_BUILD_TESTS=OFF ^
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
nmake
nmake install
```

protobuf 会安装在 build-vs2017/install 里头

## 编译安装 ncnn

https://github.com/Tencent/ncnn.git
(optional) Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
cmake 命令中的 protobuf 路径要相应修改成自己的

```batch
mkdir build-vs2017
cd build-vs2017
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install ^
    -DProtobuf_INCLUDE_DIR=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2017/install/include ^
    -DProtobuf_LIBRARIES=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2017/install/lib/libprotobuf.lib ^
    -DProtobuf_PROTOC_EXECUTABLE=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2017/install/bin/protoc.exe ..
nmake
nmake install
```

## Compile visionEngine for windows
```
Clone: git clone http://192.168.17.218:8080/tfs/DefaultCollection/Erow/_git/Cpp-VisionEigen
```

# How to use
## 1. download the models from baiduyun: [baidu](https://pan.baidu.com/s/1WguBm9JBUDEszCEi3W7E0A)(code: 8mhn) 
## 2. put models to directory: VisionEngine/data/models
## 3. https://github.com/Tencent/ncnn/releases
* Download ncnn-YYYYMMDD-platform-vulkan.zip or build ncnn for your platform yourself
* Extract ncnn-YYYYMMDD-platform-vulkan.zip into **lib/** and change the **NCNN_PATH** path to yours in **scripts/win/build_android.bat**
## 4. https://github.com/nihui/opencv-mobile
* Download opencv-mobile-XYZ-platform.zip
* Extract opencv-mobile-XYZ-platform.zip into **lib/** and change the **OPENCV_PATH** path to yours in **scripts/win/build_android.bat**
## 5. compile the project and enjoy!
```
>> cd scripts/win/ && build_android.bat
```