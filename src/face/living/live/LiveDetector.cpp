#include "LiveDetector.h"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <ncnn/net.h>


namespace mirror {
    LiveDetector::LiveDetector(FaceAntiSpoofingType type) : FaceAntiSpoofing(type) {
        //Live detection configs
        faceLivingThreshold_ = 0.93;
        struct ModelConfig config1 = {2.7f, 0.0f, 0.0f,
                                      80, 80, "model_1", false};
        struct ModelConfig config2 = {4.0f, 0.0f, 0.0f,
                                      80, 80, "model_2", false};
        configs_.clear();
        configs_.emplace_back(config1);
        configs_.emplace_back(config2);
        inputSize_.width = 80;
        inputSize_.height = 80;
    }

    int LiveDetector::loadModel(const char *root_path) {
        std::string sub_dir = "/living/live/";
        for (std::size_t i = 0; i < nets_.size(); ++i) {
            std::string model_param = std::string(root_path) + sub_dir + configs_[i].name + ".param";
            std::string model_bin = std::string(root_path) + sub_dir + configs_[i].name + ".bin";
            if (!nets_[i]) return ErrorCode::NULL_ERROR;

            if (nets_[i]->load_param(model_param.c_str()) == -1 ||
                nets_[i]->load_model(model_bin.c_str()) == -1) {
                return ErrorCode::MODEL_LOAD_ERROR;
            }
        }
        return 0;
    }

#if defined __ANDROID__
    int LiveDetector::loadModel(AAssetManager *mgr) {
        std::string sub_dir = "models/living/live/";
        for (std::size_t i = 0; i < nets_.size(); ++i) {
            std::string model_param = sub_dir + configs_[i].name + ".param";
            std::string model_bin = sub_dir + configs_[i].name + ".bin";
            if (!nets_[i]) return ErrorCode::NULL_ERROR;

            if (nets_[i]->load_param(mgr, model_param.c_str()) == -1 ||
                nets_[i]->load_model(mgr, model_bin.c_str()) == -1) {
                return ErrorCode::MODEL_LOAD_ERROR;
            }
        }
        return 0;
    }
#endif

    float LiveDetector::detectLiving(const cv::Mat &src, const cv::Rect &box) const {
        float confidence = 0.f;//score

        for (int i = 0; i < model_num_; i++) {
            cv::Mat roi;
            if (configs_[i].org_resize) {
                cv::resize(src, roi, inputSize_, 0, 0, 3);
            } else {
                cv::Rect rect = FaceAntiSpoofing::CalculateBox(box, src.cols, src.rows, configs_[i]);
                cv::resize(src(rect), roi, cv::Size(configs_[i].width, configs_[i].height));
            }

            ncnn::Mat in = ncnn::Mat::from_pixels(roi.data, ncnn::Mat::PIXEL_BGR, roi.cols, roi.rows);

            ncnn::Extractor extractor = nets_[i]->create_extractor();
            extractor.set_light_mode(true);

            extractor.input(net_input_name_.c_str(), in);
            ncnn::Mat out;
            extractor.extract(net_output_name_.c_str(), out);

            confidence += out.row(0)[1];

        }
        confidence /= model_num_;

        return confidence;
    }
}