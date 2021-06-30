#include "PoseEngine.h"
#include "detectors/PoseDetector.h"
#include "../common/Singleton.h"

#include <string>
#include <iostream>

namespace mirror {
    class PoseEngine::Impl {
    public:
        Impl() {
            initialized_ = false;
        }

        ~Impl() {
            destroyPoseDetector();
        }

        void destroyPoseDetector() {
            if (pose_detector_) {
                delete pose_detector_;
                pose_detector_ = nullptr;
            }
        }

        inline const std::vector<std::pair<int, int>> &GetJointPairs() const {
            if (!initialized_ || !pose_detector_) {
                std::cout << "pose detector model uninitialized!" << std::endl;
                return std::vector<std::pair<int, int>>();
            }
            return pose_detector_->getJointPairs();
        }

        inline int LoadModel(const PoseEigenParams &params) {
            if (pose_detector_ && pose_detector_->getType() != params.poseEstimationType) {
                destroyPoseDetector();
            }

            if (!pose_detector_) {
                switch (params.poseEstimationType) {
                    case SIMPLE_POSE:
                        pose_detector_ = SimplePoseFactory().createDetector();
                        break;
                    case LIGHT_OPEN_POSE:
                        pose_detector_ = LightOpenPoseFactory().createDetector();
                        break;
                }

                if (!pose_detector_ || pose_detector_->load(params) != 0) {
                    std::cout << "load pose detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (pose_detector_->update(params) != 0) {
                std::cout << "update pose detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            initialized_ = true;
            return 0;
        }

        inline int Detect(const cv::Mat &img_src, std::vector<PoseResult> &poses) const {
            if (!initialized_ || !pose_detector_) {
                std::cout << "pose detector model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return pose_detector_->detect(img_src, poses);
        }

    private:
        PoseDetector *pose_detector_ = nullptr;
        bool initialized_;
    };


    //! Unique instance of ecvOptions
    static Singleton<PoseEngine> s_engine;

    PoseEngine *PoseEngine::GetInstancePtr() {
        if (!s_engine.instance) {
            s_engine.instance = new PoseEngine();
        }
        return s_engine.instance;
    }

    PoseEngine &PoseEngine::GetInstance() {
        if (!s_engine.instance) {
            s_engine.instance = new PoseEngine();
        }

        return *s_engine.instance;
    }

    void PoseEngine::ReleaseInstance() {
        s_engine.release();
    }

    PoseEngine::PoseEngine() {
        impl_ = new PoseEngine::Impl();
    }

    void PoseEngine::destroyEngine() {
        ReleaseInstance();
    }

    PoseEngine::~PoseEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int PoseEngine::loadModel(const PoseEigenParams &params) {
        return impl_->LoadModel(params);
    }

    int PoseEngine::detect(const cv::Mat &img_src, std::vector<PoseResult> &poses) const {
        return impl_->Detect(img_src, poses);
    }

    const std::vector<std::pair<int, int>> &PoseEngine::getJointPairs() const {
        return impl_->GetJointPairs();
    }

}

