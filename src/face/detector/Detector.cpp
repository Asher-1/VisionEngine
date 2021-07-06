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
            scoreThreshold_(0.5f),
            modelPath_("/face/detectors") {
    }

    Detector::~Detector() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int Detector::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int Detector::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int Detector::load(const FaceEngineParams &params) {
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
            } else {
                faces.clear();
                faces.insert(faces.begin(), faces_tmp.begin(), faces_tmp.end());
            }

            if (verbose_) {
                std::cout << faces.size() << " faces detected." << std::endl;
                std::cout << "end face detect." << std::endl;
            }
        }
        return flag;
    }

    int Detector::update(const FaceEngineParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }

        // update if given
        if (params.nmsThreshold > 0) {
            iouThreshold_ = params.nmsThreshold;
        }

        // update if given
        if (params.scoreThreshold > 0) {
            scoreThreshold_ = params.scoreThreshold;
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
