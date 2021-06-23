#pragma once

#include "../ObjectDetector.h"

namespace mirror {
    class MobilenetSSD : public ObjectDetector {
    public:
        explicit MobilenetSSD(ObjectDetectorType type = ObjectDetectorType::MOBILENET_SSD);

        ~MobilenetSSD() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const override;

    private:
        const float meanVals[3] = {0.5f, 0.5f, 0.5f};
        const float normVals[3] = {0.007843f, 0.007843f, 0.007843f};
    };

}