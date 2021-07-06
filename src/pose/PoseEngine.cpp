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

        inline void PrintConfigurations(const PoseEngineParams &params) const {
            std::cout << "--------------Pose Configuration--------------" << std::endl;
            std::string configureInfo;
            configureInfo += std::string("GPU: ") + (params.gpuEnabled ? "True" : "False");
            configureInfo += std::string("\nVerbose: ") + (params.verbose ? "True" : "False");
            configureInfo += "\nModel path: " + params.modelPath;

            configureInfo += std::string("\nthread number: ") + std::to_string(params.threadNum);

            if (pose_detector_) {
                configureInfo += "\nPose detector type: " + GetPoseEstimationTypeName(pose_detector_->getType());
            }

            std::cout << configureInfo << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
        }

        inline int LoadModel(const PoseEngineParams &params) {
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

                if (!pose_detector_ || pose_detector_->load(params) != ErrorCode::SUCCESS) {
                    std::cout << "load pose detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (pose_detector_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update pose detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        inline int UpdateModel(const PoseEngineParams &params) {
            if (!pose_detector_ || pose_detector_->getType() != params.poseEstimationType) {
                return LoadModel(params);
            }

            if (pose_detector_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update pose detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
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

    int PoseEngine::loadModel(const PoseEngineParams &params) {
        return impl_->LoadModel(params);
    }

    int PoseEngine::updateModel(const PoseEngineParams &params) {
        return impl_->UpdateModel(params);
    }

    int PoseEngine::detect(const cv::Mat &img_src, std::vector<PoseResult> &poses) const {
        return impl_->Detect(img_src, poses);
    }

    const std::vector<std::pair<int, int>> &PoseEngine::getJointPairs() const {
        return impl_->GetJointPairs();
    }

}

