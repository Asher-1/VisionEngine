#pragma once

#include "../ObjectDetector.h"

namespace mirror {
    class YoloV5 : public ObjectDetector {
    public:
        explicit YoloV5(ObjectDetectorType type = ObjectDetectorType::YOLOV5);

        ~YoloV5() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const override;

    private:
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    };

}