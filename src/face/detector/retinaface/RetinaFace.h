#pragma once

#include "../Detector.h"

namespace mirror {
    class RetinaFace : public Detector {
    public:
        explicit RetinaFace(FaceDetectorType type = FaceDetectorType::RETINA_FACE);

        ~RetinaFace() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const override;

    private:
        const cv::Size inputSize_ = {640, 640};
    };

}