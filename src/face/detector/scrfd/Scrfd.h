#pragma once

#include "../Detector.h"

namespace mirror {
    class Scrfd : public Detector {
    public:
        explicit Scrfd(FaceDetectorType type = FaceDetectorType::SCRFD_FACE);

        ~Scrfd() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const override;

    private:
        const cv::Size inputSize_ = {640, 640};
        const float mean_vals_[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals_[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};

    };
}