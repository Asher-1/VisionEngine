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
            inputSize_(cv::Size(224, 224)),
            modelPath_("/classifiers") {
        class_names_.clear();
    }

    Classifier::~Classifier() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int Classifier::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int Classifier::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int Classifier::loadLabels(const char *label_path) {
        FILE *fp = fopen(label_path, "r");
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

    int Classifier::load(const ClassifierEigenParams &params) {
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

        ncnn::Option opt;

#if defined __ANDROID__
        opt.lightmode = true;
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

    int Classifier::update(const ClassifierEigenParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }

        if (params.topK > 0) {
            topk_ = params.topK;
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