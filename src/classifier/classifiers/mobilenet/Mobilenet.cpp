#define _CRT_SECURE_NO_WARNINGS

#include "Mobilenet.h"
#include <algorithm>
#include <string>
#include <ncnn/net.h>

namespace mirror {
    Mobilenet::Mobilenet(ClassifierType type) : Classifier(type) {
        topk_ = 3;
        inputSize_ = cv::Size(224, 224);
    }

    int Mobilenet::loadModel(const char *root_path) {
        std::string root_dir = std::string(root_path) + modelPath_ + "/mobilenet";
        std::string param_file = root_dir + "/mobilenet.param";
        std::string model_file = root_dir + "/mobilenet.bin";
        std::string label_file = root_dir + "/label.txt";
        if (Super::loadModel(param_file.c_str(), model_file.c_str()) != 0 ||
            Super::loadLabels(label_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int Mobilenet::loadModel(AAssetManager *mgr) {
        std::string root_dir = "models" + modelPath_ + "/mobilenet";
        std::string param_file = root_dir + "/mobilenet.param";
        std::string model_file = root_dir + "/mobilenet.bin";
        std::string label_file = root_dir + "/label.txt";
        if (Super::loadModel(mgr, param_file.c_str(), model_file.c_str()) != 0 ||
            Super::loadLabels(mgr, label_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int Mobilenet::classifyObject(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_src.cols, img_src.rows,
                                                     inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("prob", out);

        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < out.w; ++i) {
            scores.emplace_back(out[i], i);
        }

        std::partial_sort(scores.begin(), scores.begin() + topk_, scores.end(),
                          std::greater<std::pair<float, int> >());

        for (int i = 0; i < topk_; ++i) {
            ImageInfo image_info;
            image_info.label_ = class_names_[scores[i].second];
            image_info.score_ = scores[i].first;
            images.push_back(image_info);
        }
        return 0;
    }

}


