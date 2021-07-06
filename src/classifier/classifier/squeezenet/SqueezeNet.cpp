#define _CRT_SECURE_NO_WARNINGS

#include "SqueezeNet.h"
#include <string>
#include <algorithm>
#include <ncnn/net.h>

#include "squeezenet_v1.1.id.h"

namespace mirror {
    SqueezeNet::SqueezeNet(ClassifierType type) : Classifier(type) {
        topk_ = 3;
        inputSize_ = cv::Size(227, 227);
    }

    int SqueezeNet::loadModel(const char *root_path) {
        std::string root_dir = std::string(root_path) + modelPath_ + "/squeezenet";
        std::string param_file = root_dir + "/squeezenet_v1.1.param";
        std::string model_file = root_dir + "/squeezenet_v1.1.bin";
        std::string label_file = root_dir + "/synset_words.txt";
        if (Super::loadModel(param_file.c_str(), model_file.c_str()) != 0 ||
            Super::loadLabels(label_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int SqueezeNet::loadModel(AAssetManager *mgr) {
        std::string root_dir = "models" + modelPath_ + "/squeezenet";
        std::string param_file = root_dir + "/squeezenet_v1.1.param";
        std::string model_file = root_dir + "/squeezenet_v1.1.bin";
        std::string label_file = root_dir + "/synset_words.txt";
        if (Super::loadModel(mgr, param_file.c_str(), model_file.c_str()) != 0 ||
            Super::loadLabels(mgr, label_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int SqueezeNet::classifyObject(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR,
                                                     img_src.cols, img_src.rows,
                                                     inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, nullptr);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ex.input(squeezenet_v1_1_param_id::BLOB_data, in);

        ncnn::Mat out;
        ex.extract(squeezenet_v1_1_param_id::BLOB_prob, out);

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


