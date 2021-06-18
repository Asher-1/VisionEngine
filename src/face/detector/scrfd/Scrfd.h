#pragma once

#include "../Detector.h"

namespace mirror {
    class Scrfd : public Detector {
    public:
        explicit Scrfd(FaceDetectorType type = FaceDetectorType::SCRFD_FACE);

        ~Scrfd() override = default;

    protected:
        int loadModel(const char *root_path) override;

        int detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const override;

    private:
        const cv::Size inputSize_ = {640, 640};

    };
}