#pragma once

#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {
    class PoseDetector {
    public:
        using Super = PoseDetector;

        explicit PoseDetector(PoseEstimationType type);

        virtual ~PoseDetector();

        int load(const PoseEngineParams &params);

        int update(const PoseEngineParams &params);

        int detect(const cv::Mat &img_src, std::vector<PoseResult> &poses) const;

        inline PoseEstimationType getType() const { return type_; }

        inline const std::vector<std::pair<int, int>> &getJointPairs() const { return joint_pairs_; }

    protected:

        int loadModel(const char *params, const char *models);

#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
        int loadModel(AAssetManager* mgr, const char* params, const char* models);
#endif

        virtual int loadModel(const char *root_path) = 0;

        virtual int detectPose(const cv::Mat &img_src, std::vector<PoseResult> &poses) const = 0;

    protected:
        PoseEstimationType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        cv::Size inputSize_ = {640, 640};
        std::string modelPath_;
        std::vector<std::pair<int, int>> joint_pairs_;
    };

    class PoseDetectorFactory {
    public:
        virtual PoseDetector *createDetector() const = 0;

        virtual ~PoseDetectorFactory() = default;
    };

    class SimplePoseFactory : public PoseDetectorFactory {
    public:
        SimplePoseFactory() = default;

        PoseDetector *createDetector() const override;

        ~SimplePoseFactory() override = default;
    };

    class LightOpenPoseFactory : public PoseDetectorFactory {
    public:
        LightOpenPoseFactory() = default;

        PoseDetector *createDetector() const override;

        ~LightOpenPoseFactory() override = default;
    };

}