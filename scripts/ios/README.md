# VisionEngine

## Build for MacOS
### Install xcode and protobuf
```
# Install protobuf via homebrew
brew install protobuf
```

### Download and install openmp for multithreading inference feature
```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar -xf openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src

# apply some compilation fix
sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S

mkdir -p build-x86_64
cd build-x86_64
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DCMAKE_OSX_ARCHITECTURES="x86_64" \
    -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..
cmake --build . -j 4
cmake --build . --target install
cd ..

mkdir -p build-arm64
cd build-arm64
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..
cmake --build . -j 4
cmake --build . --target install
cd ..

lipo -create build-x86_64/install/lib/libomp.a build-arm64/install/lib/libomp.a -o libomp.a

# copy openmp library and header files to xcode toolchain sysroot
sudo cp build-x86_64/install/include/* /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
sudo cp libomp.a /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib
```

### Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
```
wget https://sdk.lunarg.com/sdk/download/1.2.162.0/mac/vulkansdk-macos-1.2.162.0.dmg?Human=true -O vulkansdk-macos-1.2.162.0.dmg
hdiutil attach vulkansdk-macos-1.2.162.0.dmg
cp -r /Volumes/vulkansdk-macos-1.2.162.0 .
hdiutil detach /Volumes/vulkansdk-macos-1.2.162.0

# setup env
export VULKAN_SDK=`pwd`/vulkansdk-macos-1.2.162.0/macOS

cd <ncnn-root-dir>
mkdir -p build
cd build

cmake -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib/libomp.a" \
    -DVulkan_INCLUDE_DIR=`pwd`/../vulkansdk-macos-1.2.162.0/MoltenVK/include \
    -DVulkan_LIBRARY=`pwd`/../vulkansdk-macos-1.2.162.0/MoltenVK/dylib/macOS/libMoltenVK.dylib \
    -DNCNN_VULKAN=ON ..

cmake --build . -j 4
cmake --build . --target install

```


## Build for iOS on MacOS with xcode
### Download and install openmp for multithreading inference feature on iPhoneOS
```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar -xf openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src

# apply some compilation fix
sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S

mkdir -p build-ios
cd build-ios

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install \
    -DIOS_PLATFORM=OS -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 -DIOS_ARCH="armv7;arm64;arm64e" \
    -DPERL_EXECUTABLE=/usr/local/bin/perl \
    -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..

cmake --build . -j 4
cmake --build . --target install

# copy openmp library and header files to xcode toolchain sysroot
sudo cp install/include/* /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/include
sudo cp install/lib/libomp.a /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib

```

### Download and install openmp for multithreading inference feature on iPhoneSimulator
```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar -xf openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src

# apply some compilation fix
sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S

mkdir -p build-ios-sim
cd build-ios-sim

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install \
    -DIOS_PLATFORM=SIMULATOR -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 -DIOS_ARCH="i386;x86_64" \
    -DPERL_EXECUTABLE=/usr/local/bin/perl \
    -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..

cmake --build . -j 4
cmake --build . --target install

# copy openmp library and header files to xcode toolchain sysroot
sudo cp install/include/* /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/include
sudo cp install/lib/libomp.a /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib

```

### Package openmp framework:
```
cd <openmp-root-dir>

mkdir -p openmp.framework/Versions/A/Headers
mkdir -p openmp.framework/Versions/A/Resources
ln -s A openmp.framework/Versions/Current
ln -s Versions/Current/Headers openmp.framework/Headers
ln -s Versions/Current/Resources openmp.framework/Resources
ln -s Versions/Current/openmp openmp.framework/openmp
lipo -create build-ios/install/lib/libomp.a build-ios-sim/install/lib/libomp.a -o openmp.framework/Versions/A/openmp
cp -r build-ios/install/include/* openmp.framework/Versions/A/Headers/
sed -e 's/__NAME__/openmp/g' -e 's/__IDENTIFIER__/org.llvm.openmp/g' -e 's/__VERSION__/11.0/g' Info.plist > openmp.framework/Versions/A/Resources/Info.plist

```

### Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
```
wget https://sdk.lunarg.com/sdk/download/1.2.162.0/mac/vulkansdk-macos-1.2.162.0.dmg?Human=true -O vulkansdk-macos-1.2.162.0.dmg
hdiutil attach vulkansdk-macos-1.2.162.0.dmg
cp -r /Volumes/vulkansdk-macos-1.2.162.0 .
hdiutil detach /Volumes/vulkansdk-macos-1.2.162.0

# setup env
export VULKAN_SDK=`pwd`/vulkansdk-macos-1.2.162.0/macOS

```

### Build library for iPhoneOS:
```
cd <ncnn-root-dir>
mkdir -p build-ios
cd build-ios

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS -DIOS_ARCH="armv7;arm64;arm64e" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a" \
    -DNCNN_BUILD_BENCHMARK=OFF ..

# vulkan is only available on arm64 devices
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS64 -DIOS_ARCH="arm64;arm64e" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a" \
    -DVulkan_INCLUDE_DIR=`pwd`/../vulkansdk-macos-1.2.162.0/MoltenVK/include \
    -DVulkan_LIBRARY=`pwd`/../vulkansdk-macos-1.2.162.0/MoltenVK/dylib/iOS/libMoltenVK.dylib \
    -DNCNN_VULKAN=ON -DNCNN_BUILD_BENCHMARK=OFF ..

cmake --build . -j 4
cmake --build . --target install

```

### Build library for iPhoneSimulator:
```
cd <ncnn-root-dir>
mkdir -p build-ios-sim
cd build-ios-sim

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=SIMULATOR -DIOS_ARCH="i386;x86_64" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib/libomp.a" \
    -DNCNN_BUILD_BENCHMARK=OFF ..

cmake --build . -j 4
cmake --build . --target install

```

### Package glslang framework:
```
cd <ncnn-root-dir>

mkdir -p glslang.framework/Versions/A/Headers
mkdir -p glslang.framework/Versions/A/Resources
ln -s A glslang.framework/Versions/Current
ln -s Versions/Current/Headers glslang.framework/Headers
ln -s Versions/Current/Resources glslang.framework/Resources
ln -s Versions/Current/glslang glslang.framework/glslang
libtool -static build-ios/install/lib/libglslang.a build-ios/install/lib/libSPIRV.a build-ios/install/lib/libOGLCompiler.a build-ios/install/lib/libOSDependent.a -o build-ios/install/lib/libglslang_combined.a
libtool -static build-ios-sim/install/lib/libglslang.a build-ios-sim/install/lib/libSPIRV.a build-ios-sim/install/lib/libOGLCompiler.a build-ios-sim/install/lib/libOSDependent.a -o build-ios-sim/install/lib/libglslang_combined.a
lipo -create build-ios/install/lib/libglslang_combined.a build-ios-sim/install/lib/libglslang_combined.a -o glslang.framework/Versions/A/glslang
cp -r build/install/include/glslang glslang.framework/Versions/A/Headers/
sed -e 's/__NAME__/glslang/g' -e 's/__IDENTIFIER__/org.khronos.glslang/g' -e 's/__VERSION__/1.0/g' Info.plist > glslang.framework/Versions/A/Resources/Info.plist

```


## Compile visionEngine for ios
```
Clone: git clone 
```
# How to use
## 1. download the models from baiduyun: [baidu](https://pan.baidu.com/s/1WguBm9JBUDEszCEi3W7E0A)(code: 8mhn) 
## 2. put models to directory: VisionEngine/data/models
## 3. https://github.com/Tencent/ncnn/releases
* Download ncnn-YYYYMMDD-platform-vulkan.zip or build ncnn for your platform yourself
* Extract ncnn-YYYYMMDD-platform-vulkan.zip into **lib/** and change the **NCNN_PATH** path to yours in **scripts/android/android_compile_batch.sh**
## 4. https://github.com/nihui/opencv-mobile
* Download opencv-mobile-XYZ-platform.zip
* Extract opencv-mobile-XYZ-platform.zip into **lib/** and change the **OPENCV_PATH** path to yours in **scripts/android/android_compile_batch.sh**
## 5. compile the project and enjoy!
```
>> cd scripts/ios/ && ./ ios_compile_batch.sh
```