#include "TextDetector.h"
#include "dbnet/DBNet.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace mirror {
    TextDetector::TextDetector(TextDetectorType type) :
            type_(type),
            net_(new ncnn::Net()),
            topk_(3),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            inputSize_(cv::Size(224, 224)),
            modelPath_("/ocr/detectors") {
        class_names_.clear();
    }

    TextDetector::~TextDetector() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int TextDetector::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int TextDetector::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int TextDetector::load(const OcrEngineParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;

        // update if given
        if (params.nmsThreshold > 0) {
            nmsThreshold_ = params.nmsThreshold;
        }
        // update if given
        if (params.scoreThreshold > 0) {
            scoreThreshold_ = params.scoreThreshold;
        }

        if (verbose_) {
            std::cout << "start load text detector model: "
                      << GetTextDetectorTypeName(this->type_) << std::endl;
        }

        this->net_->clear();

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
            std::cout << "load text detector model: " <<
                      GetTextDetectorTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load text detector model." << std::endl;
            }
        }
        return flag;
    }

    int TextDetector::update(const OcrEngineParams &params) {
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

    int TextDetector::detect(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const {
        textBoxes.clear();
        if (!initialized_) {
            std::cout << "text detector model: "
                      << GetTextDetectorTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start text detection." << std::endl;
        }

        int flag = this->detectText(img_src, textBoxes);
        if (flag != 0) {
            std::cout << "text detection failed." << std::endl;
        } else {
            if (verbose_) {
                std::cout << "end text detection." << std::endl;
            }
        }
        return flag;
    }

    TextDetector *DBNetFactory::create() const {
        return new DBNet();
    }
}