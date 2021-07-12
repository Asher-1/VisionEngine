#pragma once

#include "../TextDetector.h"

namespace mirror {

    class DBNet : public TextDetector {
    public:
        explicit DBNet(TextDetectorType type = TextDetectorType::DB_NET);

        ~DBNet() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *root_path) override;

        int detectText(const cv::Mat &img_src, std::vector<TextBox> &textBox) const override;

    private:
        const float meanVals[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
        const float normVals[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};
        const int min_size = 3;
        const float binThresh = 0.3;
        const float unclip_ratio = 2.0;
    };

}