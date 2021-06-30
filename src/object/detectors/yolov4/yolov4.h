#pragma once

#include "../ObjectDetector.h"

namespace mirror {
    class YoloV4 : public ObjectDetector {
    public:
        explicit YoloV4(ObjectDetectorType type = ObjectDetectorType::YOLOV4);

        ~YoloV4() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const override;

    private:
        const float meanVals[3] = {0, 0, 0};
        const float normVals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    };

}