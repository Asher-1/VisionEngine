#include "yolov4.h"

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include <ncnn/net.h>

namespace mirror {
    YoloV4::YoloV4(ObjectDetectorType type) : ObjectDetector(type) {
        scoreThreshold_ = 0.3f;
        nmsThreshold_ = 0.45f;
        inputSize_ = cv::Size(320, 320);
        class_names_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    }

    int YoloV4::loadModel(const char *root_path) {
        std::string subdir = std::string(root_path) + modelPath_ + "/yolov4/";
        std::string obj_param;
        std::string obj_bin;
        if (modeType_ == 0) {
            obj_param = subdir + "yolov4-tiny-opt.param";
            obj_bin = subdir + "yolov4-tiny-opt.bin";
        } else if (modeType_ == 1) {
            obj_param = subdir + "MobileNetV2-YOLOv3-Nano-coco.param";
            obj_bin = subdir + "MobileNetV2-YOLOv3-Nano-coco.bin";
        } else if (modeType_ == 2) {
            obj_param = subdir + "yolo-fastest-opt.param";
            obj_bin = subdir + "yolo-fastest-opt.bin";
        } else {
            obj_param = subdir + "yolov4-tiny-opt.param";
            obj_bin = subdir + "yolov4-tiny-opt.bin";
        }

        return Super::loadModel(obj_param.c_str(), obj_bin.c_str());
    }

#if defined __ANDROID__
    int YoloV4::loadModel(AAssetManager *mgr) {
        std::string subdir = "models" + modelPath_ + "/yolov4/";
        std::string obj_param;
        std::string obj_bin;
        if (modeType_ == 0) {
            obj_param = subdir + "yolov4-tiny-opt.param";
            obj_bin = subdir + "yolov4-tiny-opt.bin";
        } else if (modeType_ == 1) {
            obj_param = subdir + "MobileNetV2-YOLOv3-Nano-coco.param";
            obj_bin = subdir + "MobileNetV2-YOLOv3-Nano-coco.bin";
        } else if (modeType_ == 2) {
            obj_param = subdir + "yolo-fastest-opt.param";
            obj_bin = subdir + "yolo-fastest-opt.bin";
        } else {
            obj_param = subdir + "yolov4-tiny-opt.param";
            obj_bin = subdir + "yolov4-tiny-opt.bin";
        }

        return Super::loadModel(mgr, obj_param.c_str(), obj_bin.c_str());
    }
#endif

    int YoloV4::detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        int img_width = img_src.cols;
        int img_height = img_src.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB, img_width,
                                                     img_height, inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = net_->create_extractor();
        if (this->gpu_mode_) {  // 消除提示
            ex.set_vulkan_compute(this->gpu_mode_);
        }

        ex.input(0, in);
        ncnn::Mat blob;
        ex.extract("output", blob);

        objects.clear();
        for (int i = 0; i < blob.h; i++) {
            const float *values = blob.row(i);
            if (values[1] < scoreThreshold_) {
                continue;
            }

            ObjectInfo object;
            object.name_ = class_names_[int(values[0]) - 1];
            object.score_ = values[1];
            object.location_.x = static_cast<int>(values[2] * img_width);
            object.location_.y = static_cast<int>(values[3] * img_height);
            object.location_.width = static_cast<int>(values[4] * img_width - object.location_.x);
            object.location_.height = static_cast<int>(values[5] * img_height - object.location_.y);
            objects.push_back(object);
        }

        return 0;
    }

}