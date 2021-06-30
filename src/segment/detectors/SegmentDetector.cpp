#include "SegmentDetector.h"
#include "mobilenetv3/MobileNetV3Seg.h"
#include "yolact/Yolact.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace mirror {
    SegmentDetector::SegmentDetector(SegmentType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            scoreThreshold_(0.7f),
            nmsThreshold_(0.5f),
            inputSize_(cv::Size(550, 550)),
            modelPath_("/segment") {
        class_names_.clear();
    }

    SegmentDetector::~SegmentDetector() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int SegmentDetector::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int SegmentDetector::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int SegmentDetector::load(const SegmentEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;
        if (verbose_) {
            std::cout << "start load segment detector model: "
                      << GetSegmentTypeName(this->type_) << std::endl;
        }

        // update if given
        if (params.nmsThreshold > 0) {
            nmsThreshold_ = params.nmsThreshold;
        }
        // update if given
        if (params.scoreThreshold > 0) {
            scoreThreshold_ = params.scoreThreshold;
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
            std::cout << "load segment detector model: " <<
                      GetSegmentTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load segment detector model." << std::endl;
            }
        }
        return flag;
    }

    int SegmentDetector::update(const SegmentEigenParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }

        // update if given
        if (params.nmsThreshold > 0) {
            nmsThreshold_ = params.nmsThreshold;
        }
        // update if given
        if (params.scoreThreshold > 0) {
            scoreThreshold_ = params.scoreThreshold;
        }

        return flag;
    }


    int SegmentDetector::detect(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const {
        segments.clear();
        if (!initialized_) {
            std::cout << "segment detector model: "
                      << GetSegmentTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start segment detect." << std::endl;
        }

        int flag = this->detectSeg(img_src, segments);
        if (flag != 0) {
            std::cout << "segment detect failed." << std::endl;
        } else {
            if (verbose_) {
                std::cout << "end segment detect." << std::endl;
            }
        }
        return flag;
    }

    SegmentDetector *YolactSegFactory::createDetector() const {
        return new Yolact();
    }

    SegmentDetector *MobileNetV3SegFactory::createDetector() const {
        return new MobileNetV3Seg();
    }
}