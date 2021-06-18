#include "Detector.h"
#include "mtcnn/MtcnnFace.h"
#include "centerface/CenterFace.h"
#include "retinaface/RetinaFace.h"
#include "anticov/AntiCovFace.h"
#include "scrfd/Scrfd.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>

#include <iostream>

namespace mirror {
    Detector::Detector(FaceDetectorType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            has_kps_(true),
            iouThreshold_(0.45f),
            scoreThreshold_(0.5f) {
    }

    Detector::~Detector() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int Detector::load(const char *root_path, const FaceEigenParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;
        // update if given
        if (params.nmsThreshold > 0) {
            iouThreshold_ = params.nmsThreshold;
        }
        // update if given
        if (params.scoreThreshold > 0) {
            scoreThreshold_ = params.scoreThreshold;
        }

        if (verbose_) {
            std::cout << "start load detector model: " << GetDetectorTypeName(this->type_) << std::endl;
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
            std::cout << "load detector model: " << GetDetectorTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load detector model!" << std::endl;
            }
        }
        return flag;
    }

    int Detector::detect(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const {
        faces.clear();
        if (!initialized_) {
            std::cout << "face detector model: "
                      << GetDetectorTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }
        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start detect." << std::endl;
        }

        std::vector<FaceInfo> faces_tmp;
        int flag = this->detectFace(img_src, faces_tmp);
        if (flag != 0) {
            std::cout << "detect failed." << std::endl;
        } else {
            // mtcnn faces have been nms processed internally!
            if (this->type_ != FaceDetectorType::MTCNN_FACE) {
                NMS(faces_tmp, faces, iouThreshold_);
            }

            if (verbose_) {
                std::cout << faces.size() << " faces detected." << std::endl;
                std::cout << "end face detect." << std::endl;
            }
        }
        return flag;
    }

    Detector *CenterfaceFactory::CreateDetector() const {
        return new CenterFace();
    }

    Detector *MtcnnFactory::CreateDetector() const {
        return new MtcnnFace();
    }

    Detector *RetinafaceFactory::CreateDetector() const {
        return new RetinaFace();
    }

    Detector *AnticovFactory::CreateDetector() const {
        return new AntiCovFace();
    }

    Detector *ScrfdFactory::CreateDetector() const {
        return new Scrfd();
    }

}
