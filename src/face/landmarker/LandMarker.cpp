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

    int LandMarker::load(const char *root_path, const FaceEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;

        if (verbose_) {
            std::cout << "start load landmarks model: "
                      << GetLandMarkerTypeName(this->type_) << std::endl;
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
