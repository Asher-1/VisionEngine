#pragma once

#include "../Classifier.h"

namespace mirror {

    class Mobilenet : public Classifier {
    public:
        explicit Mobilenet(ClassifierType type = ClassifierType::MOBILE_NET);

        ~Mobilenet() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *root_path) override;

        int classifyObject(const cv::Mat &img_src, std::vector<ImageInfo> &images) const override;

    private:
        const float meanVals[3] = {103.94f, 116.78f, 123.68f};
        const float normVals[3] = {0.017f, 0.017f, 0.017f};
    };

}