#include "Recognizer.h"
#include "./mobilefacenet/MobileFacenet.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>

#include <iostream>

namespace mirror {

    Recognizer::Recognizer(FaceRecognizerType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            faceFaceFeatureDim_(kFaceFeatureDim),
            inputSize_(cv::Size(112, 112)) {
    }

    Recognizer::~Recognizer() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int Recognizer::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int Recognizer::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif


    int Recognizer::load(const FaceEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;

        if (verbose_) {
            std::cout << "start load recognizer model: "
                      << GetRecognizerTypeName(this->type_) << std::endl;
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
        this->gpu_mode_ = params.gpuEnabled;
        if (ncnn::get_gpu_count() != 0) {
            opt.use_vulkan_compute = this->gpu_mode_;
        }
#endif // NCNN_VULKAN

        this->net_->opt = opt;

#if defined __ANDROID__
        int flag = this->loadModel(params.mgr);
#else
        int flag = this->loadModel(params.modelPath.c_str());
#endif
        if (flag != 0) {
            initialized_ = false;
            std::cout << "load recognizer model: " <<
                      GetRecognizerTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load recognizer model." << std::endl;
            }
        }
        return flag;
    }

    int Recognizer::update(const FaceEigenParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }
        return flag;
    }

    int Recognizer::extract(const cv::Mat &img_face, std::vector<float> &feature) const {
        feature.clear();
        if (!initialized_) {
            std::cout << "face recognizer model: "
                      << GetRecognizerTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }
        if (img_face.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start extract feature." << std::endl;
        }

        int flag = this->extractFeature(img_face, feature);
        if (flag != 0) {
            std::cout << "extract failed." << std::endl;
        } else {
            //    This is a normalize function before calculating the cosine distance. Experiment has proven it can destory the
            //    original distribution in order to make two feature more distinguishable.
            //    mean value is set to 0 and std is set to 1
            float mean;
            float variance;
            ComputeMeanAndVariance<float>(feature, mean, &variance);
            float stdDev = std::sqrt(variance);
            for (std::size_t i = 0; i < feature.size(); ++i) {
                feature.at(i) = (feature.at(i) - mean) / stdDev;
            }

            if (verbose_) {
                std::cout << "end extract feature." << std::endl;
            }
        }
        return flag;
    }

    Recognizer *MobilefacenetRecognizerFactory::CreateRecognizer() const {
        return new MobileFacenet();
    }
}
