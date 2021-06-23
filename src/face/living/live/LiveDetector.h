#pragma once

#include "../FaceAntiSpoofing.h"

namespace mirror {
    class LiveDetector : public FaceAntiSpoofing {
    public:
        LiveDetector(FaceAntiSpoofingType type = FaceAntiSpoofingType::LIVE_FACE);

        ~LiveDetector() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        float detectLiving(const cv::Mat &src, const cv::Rect &box) const override;

    private:
        const std::string net_input_name_ = "data";
        const std::string net_output_name_ = "softmax";
    };
}
