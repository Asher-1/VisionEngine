#include "PoseDetector.h"
#include "simplepose/SimplePose.h"
#include "lightopenpose/LightOpenPose.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace mirror {
    PoseDetector::PoseDetector(PoseEstimationType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            inputSize_(cv::Size(640, 640)),
            modelPath_("/pose") {
    }

    PoseDetector::~PoseDetector() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int PoseDetector::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int PoseDetector::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int PoseDetector::load(const PoseEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;
        if (verbose_) {
            std::cout << "start load pose detector model: "
                      << GetPoseEstimationTypeName(this->type_) << std::endl;
        }

        this->net_->clear();

        ncnn::Option opt;

#if defined __ANDROID__
        opt.lightmode = true;
        opt.use_fp16_arithmetic = true;
        ncnn::set_cpu_powersave(CUSTOM_THREAD_NUMBER);
#endif
        int max_thread_num = ncnn::get_big_cpu_count();
        int num_threads = max_thread_num;
        if (params.thread_num > 0 && params.thread_num < max_thread_num) {
            num_threads = params.thread_num;
        }
        ncnn::set_omp_num_threads(num_threads);
        opt.num_threads = num_threads;

#if NCNN_VULKAN
        this->gpu_mode_ = params.gpuEnabled && ncnn::get_gpu_count() > 0;
        opt.use_vulkan_compute = this->gpu_mode_;
#endif // NCNN_VULKAN

        this->net_->opt = opt;

#if defined __ANDROID__
        int flag = this->loadModel(params.mgr);
#else
        int flag = this->loadModel(params.modelPath.c_str());
#endif

        if (flag != 0) {
            initialized_ = false;
            std::cout << "load pose detector model: " <<
                      GetPoseEstimationTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load pose detector model." << std::endl;
            }
        }
        return flag;
    }

    int PoseDetector::update(const PoseEigenParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }
        return flag;
    }


    int PoseDetector::detect(const cv::Mat &img_src, std::vector<PoseResult> &poses) const {
        poses.clear();
        if (!initialized_) {
            std::cout << "pose detector model: "
                      << GetPoseEstimationTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start pose detect." << std::endl;
        }

        int flag = this->detectPose(img_src, poses);
        if (flag != 0) {
            std::cout << "pose detect failed." << std::endl;
        } else {
            if (verbose_) {
                std::cout << "end pose detect." << std::endl;
            }
        }
        return flag;
    }

    PoseDetector *SimplePoseFactory::createDetector() const {
        return new SimplePose();
    }

    PoseDetector *LightOpenPoseFactory::createDetector() const {
        return new LightOpenPose();
    }
}