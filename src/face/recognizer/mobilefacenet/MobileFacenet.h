#pragma once

#include "../Recognizer.h"

namespace mirror {

    class MobileFacenet : public Recognizer {
    public:
        using Super = Recognizer;
        explicit MobileFacenet(FaceRecognizerType type = FaceRecognizerType::ARC_FACE);

        ~MobileFacenet() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int extractFeature(const cv::Mat &img_face, std::vector<float> &feature) const override;
    };

}