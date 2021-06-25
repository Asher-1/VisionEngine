#include "LandMarker.h"
#include "zqlandmarker/ZQLandMarker.h"
#include "insightface/InsightfaceLandMarker.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>

#include <iostream>

namespace mirror {

    LandMarker::LandMarker(FaceLandMarkerType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            inputSize_(cv::Size(112, 112)) {
    }

    LandMarker::~LandMarker() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int LandMarker::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }


#if defined __ANDROID__
    int LandMarker::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int LandMarker::load(const FaceEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;

        if (verbose_) {
            std::cout << "start load landmarks model: "
                      << GetLandMarkerTypeName(this->type_) << std::endl;
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
            std::cout << "load face landmark model: " <<
                      GetLandMarkerTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load face landmark model." << std::endl;
            }
        }
        return flag;
    }

    int LandMarker::update(const FaceEigenParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }
        return flag;
    }

    int LandMarker::extract(const cv::Mat &img_src, const cv::Rect &face,
                            std::vector<cv::Point2f> &keypoints) const {
        keypoints.clear();
        if (!initialized_) {
            std::cout << "face landmark model: "
                      << GetLandMarkerTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty() || face.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start extract keypoints." << std::endl;
        }

        int flag = this->extractKeypoints(img_src, face, keypoints);

        if (flag != 0) {
            std::cout << "extract failed." << std::endl;
        } else {
            if (verbose_) {
                std::cout << "end extract keypoints." << std::endl;
            }
        }
        return flag;
    }

    LandMarker *ZQLandMarkerFactory::CreateLandmarker() const {
        return new ZQLandMarker();
    }

    LandMarker *InsightfaceLandMarkerFactory::CreateLandmarker() const {
        return new InsightfaceLandMarker();
    }
}
