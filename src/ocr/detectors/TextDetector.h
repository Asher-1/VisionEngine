#pragma once

#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {

    class TextDetector {
    public:
        using Super = TextDetector;

        explicit TextDetector(TextDetectorType type);

        virtual ~TextDetector();

        int load(const OcrEngineParams &params);

        int update(const OcrEngineParams &params);

        int detect(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const;

        inline TextDetectorType getType() const { return type_; }

    protected:

        int loadModel(const char *params, const char *models);

#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
        int loadModel(AAssetManager* mgr, const char* params, const char* models);
#endif

        virtual int loadModel(const char *root_path) = 0;
        virtual int detectText(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const = 0;

    protected:
        TextDetectorType type_;
        ncnn::Net *net_ = nullptr;
        int topk_ = 5;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        float scoreThreshold_ = 0.7f;
        float nmsThreshold_ = 0.5f;
        std::vector<std::string> class_names_;
        cv::Size inputSize_ = {224, 224};
        std::string modelPath_;
    };

    class TextDetectorFactory {
    public:
        virtual TextDetector *create() const = 0;

        virtual ~TextDetectorFactory() = default;
    };

    class DBNetFactory : public TextDetectorFactory {
    public:
        DBNetFactory() = default;

        ~DBNetFactory() override = default;

        TextDetector *create() const override;

    };
}
