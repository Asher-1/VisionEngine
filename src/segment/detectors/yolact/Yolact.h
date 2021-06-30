#pragma once

#include "../SegmentDetector.h"

namespace mirror {
    class Yolact : public SegmentDetector {
    public:
        explicit Yolact(SegmentType type = SegmentType::YOLACT_SEG);

        ~Yolact() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectSeg(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const override;

    private:
        const int keep_top_k = 200;
    };

}