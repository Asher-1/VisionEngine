#pragma once

#include "../Detector.h"

namespace mirror {
    using ANCHORS = std::vector<cv::Rect>;

    class AntiCovFace : public Detector {
    public:
        explicit AntiCovFace(FaceDetectorType type = FaceDetectorType::ANTICOV_FACE);

        ~AntiCovFace() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const override;

    private:
        const int RPNs_[3] = {32, 16, 8};
        const cv::Size inputSize_ = {640, 640};
        const float maskThreshold_ = 0.2f;
        std::vector<ANCHORS> anchors_generated_;
    };

}
