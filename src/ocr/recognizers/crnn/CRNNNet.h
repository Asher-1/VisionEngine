#pragma once

#include "../TextRecognizer.h"

namespace mirror {

    class CRNNNet : public TextRecognizer {
    public:
        explicit CRNNNet(TextRecognizerType type = TextRecognizerType::CRNN_NET);

        ~CRNNNet() override;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *root_path) override;

        int recognizeText(const cv::Mat &img_src,
                          const std::vector<TextBox> &textBoxes,
                          std::vector<OCRResult> &ocrResults) const override;

    private:
        ncnn::Net *angleNet_;
        const int crnn_h = 32;
        const float meanVals[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
        const float normVals[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};

        const float angleMeanVals[3] = {127.5, 127.5, 127.5};
        const float angleNormVals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
        const int angle_target_w = 192;
        const int angle_target_h = 32;
    };

}