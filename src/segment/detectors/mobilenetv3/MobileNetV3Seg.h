#pragma once

#include "../SegmentDetector.h"

namespace mirror {
    class MobileNetV3Seg : public SegmentDetector {
    public:
        explicit MobileNetV3Seg(SegmentType type = SegmentType::MOBILENETV3_SEG);

        ~MobileNetV3Seg() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectSeg(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const override;
    };

}