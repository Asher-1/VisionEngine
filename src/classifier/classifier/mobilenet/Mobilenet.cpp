﻿#define _CRT_SECURE_NO_WARNINGS

#include "Mobilenet.h"
#include <algorithm>
#include <string>
#include "ncnn/net.h"

namespace mirror {
    Mobilenet::Mobilenet(ClassifierType type) : Classifier(type) {
        topk_ = 5;
        inputSize_ = cv::Size(224, 224);
    }

    int Mobilenet::loadModel(const char *root_path) {
        std::string root_dir = std::string(root_path) + "/classifiers/mobilenet";
        std::string param_file = root_dir + "/mobilenet.param";
        std::string model_file = root_dir + "/mobilenet.bin";
        if (net_->load_param(param_file.c_str()) == -1 ||
            net_->load_model(model_file.c_str()) == -1 ||
            loadLabels(root_dir.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }

    int Mobilenet::classifyObject(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_src.cols, img_src.rows,
                                                     inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = net_->create_extractor();
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

    int Mobilenet::loadLabels(const char *root_path) {
        std::string label_file = std::string(root_path) + "/label.txt";
        FILE *fp = fopen(label_file.c_str(), "r");
        if (!fp) {
            return ErrorCode::NULL_ERROR;
        }

        class_names_.clear();
        while (!feof(fp)) {
            char str[1024];
            if (nullptr == fgets(str, 1024, fp)) continue;
            std::string str_s(str);

            if (str_s.length() > 0) {
                for (int i = 0; i < str_s.length(); i++) {
                    if (str_s[i] == ' ') {
                        std::string strr = str_s.substr(i, str_s.length() - i - 1);
                        class_names_.push_back(strr);
                        i = str_s.length();
                    }
                }
            }
        }
        return 0;
    }
}


