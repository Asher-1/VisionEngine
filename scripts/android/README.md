# Build for Android

This is a simple visionEngine project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile


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
>> cd scripts/android/ && ./ android_compile_batch.sh
```

## some notes
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU on Mobile platform, this is common
* FPS may be lower in dark environment because of longer camera exposure time