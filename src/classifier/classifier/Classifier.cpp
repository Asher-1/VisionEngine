#include "Classifier.h"
#include "mobilenet/Mobilenet.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace mirror {
    Classifier::Classifier(ClassifierType type) :
            type_(type),
            net_(new ncnn::Net()),
            topk_(5),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            inputSize_(cv::Size(224, 224)) {
        class_names_.clear();
    }

    Classifier::~Classifier() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int Classifier::load(const char *root_path, const ClassifierEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;
        if (params.topK > 0) {
            topk_ = params.topK;
        }

        if (verbose_) {
            std::cout << "start load classifier model: "
                      << GetClassifierTypeName(this->type_) << std::endl;
        }

        this->net_->clear();

#if defined __ANDROID__
        ncnn::set_cpu_powersave(CUSTOM_THREAD_NUMBER);
#endif
        int max_thread_num = ncnn::get_big_cpu_count();
        int num_threads = max_thread_num;
        if (params.thread_num > 0 && params.thread_num < max_thread_num) {
            num_threads = params.thread_num;
        }
        ncnn::set_omp_num_threads(num_threads);
        this->net_->opt = ncnn::Option();

#if NCNN_VULKAN
        this->gpu_mode_ = params.gpuEnabled;
        this->net_->opt.use_vulkan_compute = this->gpu_mode_;
#endif // NCNN_VULKAN

        this->net_->opt.num_threads = num_threads;
        int flag = this->loadModel(root_path);
        if (flag != 0) {
            initialized_ = false;
            std::cout << "load classifier model: " <<
                      GetClassifierTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load classifier model." << std::endl;
            }
        }
        return flag;
    }

    int Classifier::classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
        images.clear();
        if (!initialized_) {
            std::cout << "object classifier model: "
                      << GetClassifierTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start object classify." << std::endl;
        }

        int flag = this->classifyObject(img_src, images);
        if (flag != 0) {
            std::cout << "object classify failed." << std::endl;
        } else {
            if (verbose_) {
                std::cout << "this object is most likely to be: " <<
                          images[0].label_ << " (" << images[0].score_ << ")" << std::endl;
                std::cout << "end object classify." << std::endl;
            }
        }
        return flag;
    }

    Classifier *MobilenetFactory::createClassifier() const {
        return new Mobilenet();
    }

}