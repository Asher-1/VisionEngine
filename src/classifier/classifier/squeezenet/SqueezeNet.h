#pragma once

#include "../Classifier.h"

namespace mirror {

    class SqueezeNet : public Classifier {
    public:
        explicit SqueezeNet(ClassifierType type = ClassifierType::SQUEEZE_NET);

        ~SqueezeNet() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *root_path) override;

        int classifyObject(const cv::Mat &img_src, std::vector<ImageInfo> &images) const override;

    private:
        const float meanVals[3] = {104.f, 117.f, 123.f};
    };

}