#pragma once

#include "../PoseDetector.h"

namespace mirror {
    class SimplePose : public PoseDetector {
    public:
        explicit SimplePose(PoseEstimationType type = PoseEstimationType::SIMPLE_POSE);

        ~SimplePose() override;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectPose(const cv::Mat &img_src, std::vector<PoseResult> &poses) const override;

        int runPose(cv::Mat &roi, float x1, float y1, std::vector<KeyPoint> &keypoints) const;

    private:
        // for person detector
        ncnn::Net *PersonNet;
        const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        const float mean[3] = {0, 0, 0};
        const int detector_size_width = 320;
        const int detector_size_height = 320;

        // for pose detector
        const float meanVals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
        const float normVals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    };

}