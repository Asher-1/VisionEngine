#include "FaceAntiSpoofing.h"
#include "./live/LiveDetector.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>

#include <iostream>

namespace mirror {

    FaceAntiSpoofing::FaceAntiSpoofing(FaceAntiSpoofingType type) :
            type_(type),
            model_num_(1),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            faceLivingThreshold_(0.93),
            inputSize_(cv::Size(80, 80)),
            modelPath_("/face/living") {
        clearNets();
        configs_.clear();
    }

    FaceAntiSpoofing::~FaceAntiSpoofing() {
        clearNets();
    }

    void FaceAntiSpoofing::clearNets() {
        for (auto &net : nets_) {
            if (net) {
                net->clear();
                delete net;
                net = nullptr;
            }
        }
        nets_.clear();
    }

    int FaceAntiSpoofing::load(const FaceEigenParams &params) {
        verbose_ = params.verbose;
        // update if given
        if (params.livingThreshold > 0) {
            faceLivingThreshold_ = params.livingThreshold;
        }

        if (verbose_) {
            std::cout << "start load face anti spoofing model: "
                      << GetAntiSpoofingTypeName(this->type_) << std::endl;
        }

        clearNets();
        ncnn::Option opt;

#if defined __ANDROID__
        opt.lightmode = true;
        ncnn::set_cpu_powersave(CUSTOM_THREAD_NUMBER);
#endif
        int max_thread_num = ncnn::get_big_cpu_count();
        int num_threads = max_thread_num;
        if (params.threadNum > 0 && params.threadNum < max_thread_num) {
            num_threads = params.threadNum;
        }
        ncnn::set_omp_num_threads(num_threads);
        opt.num_threads = num_threads;

#if NCNN_VULKAN
        this->gpu_mode_ = params.gpuEnabled;
        if (ncnn::get_gpu_count() != 0) {
            opt.use_vulkan_compute = this->gpu_mode_;
        }
#endif // NCNN_VULKAN

        model_num_ = static_cast<int>(configs_.size());
        for (int i = 0; i < model_num_; ++i) {
            auto net = new ncnn::Net();
            net->opt = opt;
            nets_.emplace_back(net);
        }

        if (nets_.empty()) return ErrorCode::NULL_ERROR;

#if defined __ANDROID__
        int flag = this->loadModel(params.mgr);
#else
        int flag = this->loadModel(params.modelPath.c_str());
#endif
        if (flag != 0) {
            initialized_ = false;
            std::cout << "load face anti spoofing model: "
                      << GetAntiSpoofingTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load face anti spoofing model." << std::endl;
            }
        }
        return flag;
    }

    int FaceAntiSpoofing::update(const FaceEigenParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }

        // update if given
        if (params.livingThreshold > 0) {
            faceLivingThreshold_ = params.livingThreshold;
        }
        return flag;
    }

    bool FaceAntiSpoofing::detect(const cv::Mat &src, const cv::Rect &box, float &livingScore) const {
        if (!initialized_) {
            std::cout << "face anti spoofing model: "
                      << GetAntiSpoofingTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return false;
        }
        if (src.empty() || box.empty()) {
            std::cout << "input empty." << std::endl;
            return false;
        }

        if (verbose_) {
            std::cout << "start detect living face." << std::endl;
        }

        livingScore = this->detectLiving(src, box);
        if (verbose_) {
            if (livingScore >= faceLivingThreshold_) {
                std::cout << "detect living face." << std::endl;
            } else {
                std::cout << "detect anti living face." << std::endl;
            }
            std::cout << "end detect living face." << std::endl;
        }

        return livingScore >= faceLivingThreshold_;
    }

    cv::Rect FaceAntiSpoofing::CalculateBox(const cv::Rect &box, int w, int h, const ModelConfig &config) {
        int x = static_cast<int>(box.tl().x);
        int y = static_cast<int>(box.tl().y);
        int box_width = static_cast<int>(box.width);
        int box_height = static_cast<int>(box.height);

        int shift_x = static_cast<int>(box_width * config.shift_x);
        int shift_y = static_cast<int>(box_height * config.shift_y);

        float scale = std::min(config.scale, std::min((w - 1) / (float) box_width, (h - 1) / (float) box_height));

        int box_center_x = box_width / 2 + x;
        int box_center_y = box_height / 2 + y;

        int new_width = static_cast<int>(box_width * scale);
        int new_height = static_cast<int>(box_height * scale);

        int left_top_x = box_center_x - new_width / 2 + shift_x;
        int left_top_y = box_center_y - new_height / 2 + shift_y;
        int right_bottom_x = box_center_x + new_width / 2 + shift_x;
        int right_bottom_y = box_center_y + new_height / 2 + shift_y;

        if (left_top_x < 0) {
            right_bottom_x -= left_top_x;
            left_top_x = 0;
        }

        if (left_top_y < 0) {
            right_bottom_y -= left_top_y;
            left_top_y = 0;
        }

        if (right_bottom_x >= w) {
            int s = right_bottom_x - w + 1;
            left_top_x -= s;
            right_bottom_x -= s;
        }

        if (right_bottom_y >= h) {
            int s = right_bottom_y - h + 1;
            left_top_y -= s;
            right_bottom_y -= s;
        }
        return cv::Rect(left_top_x, left_top_y, new_width, new_height);
    }

    FaceAntiSpoofing *LiveDetectorFactory::CreateFaceAntiSpoofing() const {
        return new LiveDetector();
    }
}
