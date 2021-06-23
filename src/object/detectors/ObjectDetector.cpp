#include "ObjectDetector.h"
#include "yolov5/yolov5.h"
#include "mobilenetssd/MobilenetSSD.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace mirror {
    ObjectDetector::ObjectDetector(ObjectDetectorType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            scoreThreshold_(0.7f),
            nmsThreshold_(0.5f),
            inputSize_(cv::Size(640, 640)) {
        class_names_.clear();
    }

    ObjectDetector::~ObjectDetector() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }


    int ObjectDetector::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int ObjectDetector::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int ObjectDetector::load(const ObjectEigenParams &params) {
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
            std::cout << "start load object detector model: "
                      << GetObjectDetectorTypeName(this->type_) << std::endl;
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
        int flag = this->loadModel(params.model_path.c_str());
        if (flag != 0) {
            initialized_ = false;
            std::cout << "load object detector model: " <<
                      GetObjectDetectorTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load object detector model." << std::endl;
            }
        }
        return flag;
    }

    int ObjectDetector::update(const ObjectEigenParams &params) {
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


    int ObjectDetector::detect(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        objects.clear();
        if (!initialized_) {
            std::cout << "face object detector model: "
                      << GetObjectDetectorTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start object detect." << std::endl;
        }

        std::vector<ObjectInfo> objects_tmp;
        int flag = this->detectObject(img_src, objects_tmp);
        if (flag != 0) {
            std::cout << "object detect failed." << std::endl;
        } else {
            NMS(objects_tmp, objects, nmsThreshold_);
            if (verbose_) {
                std::cout << "objects number: " << objects.size() << std::endl;
                std::cout << "end object detect." << std::endl;
            }
        }
        return flag;
    }

    ObjectDetector *Yolov5Factory::createDetector() const {
        return new YoloV5();
    }

    ObjectDetector *MobilenetSSDFactory::createDetector() const {
        return new MobilenetSSD();
    }
}