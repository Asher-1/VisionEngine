#include "MobilenetSSD.h"

#include <vector>
#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#include "ncnn/net.h"

namespace mirror {
    MobilenetSSD::MobilenetSSD(ObjectDetectorType type) : ObjectDetector(type) {
        scoreThreshold_ = 0.7f;
        nmsThreshold_ = 0.5f;
        inputSize_ = cv::Size(300, 300);
        class_names_ = {
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor"};
    }

    int MobilenetSSD::loadModel(const char *root_path) {
        std::string sub_dir = "/object_detectors/mobilenetssd";
        std::string obj_param = std::string(root_path) + sub_dir + "/mobilenetssd.param";
        std::string obj_bin = std::string(root_path) + sub_dir + "/mobilenetssd.bin";
        if (Super::loadModel(obj_param.c_str(), obj_bin.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }

#if defined __ANDROID__
    int MobilenetSSD::loadModel(AAssetManager *mgr) {
        std::string sub_dir = "models/object_detectors/mobilenetssd";
        std::string obj_param = sub_dir + "/mobilenetssd.param";
        std::string obj_bin =  sub_dir + "/mobilenetssd.bin";
        if (Super::loadModel(mgr, obj_param.c_str(), obj_bin.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }
#endif

    int MobilenetSSD::detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        int width = img_src.cols;
        int height = img_src.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR, img_src.cols,
                                                     img_src.rows, inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = net_->create_extractor();
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("detection_out", out);

        objects.clear();
        for (int i = 0; i < out.h; i++) {
            const float *values = out.row(i);
            // filter the result
            if (values[1] < scoreThreshold_) {
                continue;
            }

            ObjectInfo object;
            object.name_ = class_names_[int(values[0])];
            object.score_ = values[1];
            object.location_.x = static_cast<int>(values[2] * width);
            object.location_.y = static_cast<int>(values[3] * height);
            object.location_.width = static_cast<int>(values[4] * width - object.location_.x);
            object.location_.height = static_cast<int>(values[5] * height - object.location_.y);

            objects.push_back(object);
        }
        return 0;
    }

}
