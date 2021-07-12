#pragma once

#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {

    class TextRecognizer {
    public:
        using Super = TextRecognizer;

        explicit TextRecognizer(TextRecognizerType type);

        virtual ~TextRecognizer();

        int load(const OcrEngineParams &params);

        int update(const OcrEngineParams &params);

        int recognize(const cv::Mat &img_src,
                      const std::vector<TextBox> &textBoxes,
                      std::vector<OCRResult> &ocrResults) const;

        inline TextRecognizerType getType() const { return type_; }

    protected:
        int loadLabels(const char *label_path);

        int loadModel(const char *params, const char *models);

#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
        int loadLabels(AAssetManager *mgr, const char *label_path);
        int loadModel(AAssetManager* mgr, const char* params, const char* models);
#endif

        virtual int loadModel(const char *root_path) = 0;

        virtual int recognizeText(const cv::Mat &img_src,
                                  const std::vector<TextBox> &textBoxes,
                                  std::vector<OCRResult> &ocrResults) const = 0;

    protected:
        TextRecognizerType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        int threadNum_ = 4;
        std::vector<std::string> class_names_;
        cv::Size inputSize_ = {224, 224};
        std::string modelPath_;
    };

    class TextRecognizerFactory {
    public:
        virtual TextRecognizer *create() const = 0;

        virtual ~TextRecognizerFactory() = default;
    };

    class CRNNNetFactory : public TextRecognizerFactory {
    public:
        CRNNNetFactory() = default;

        ~CRNNNetFactory() override = default;

        TextRecognizer *create() const override;

    };
}
