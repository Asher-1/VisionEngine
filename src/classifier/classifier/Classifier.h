#pragma once

#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {

    class Classifier {
    public:
        explicit Classifier(ClassifierType type);

        virtual ~Classifier();

        int load(const char *root_path, const ClassifierEigenParams& params);

        int classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const;

        inline ClassifierType getType() const { return type_; }

    protected:
        virtual int loadModel(const char *root_path) = 0;

        virtual int classifyObject(const cv::Mat &img_src, std::vector<ImageInfo> &images) const = 0;

    protected:
        ClassifierType type_;
        ncnn::Net *net_ = nullptr;
        int topk_ = 5;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        std::vector<std::string> class_names_;
        cv::Size inputSize_ = {224, 224};
    };

    class ClassifierFactory {
    public:
        virtual Classifier *createClassifier() const = 0;

        virtual ~ClassifierFactory() = default;
    };

    class MobilenetFactory : public ClassifierFactory {
    public:
        MobilenetFactory() = default;
        ~MobilenetFactory() override = default;

        Classifier *createClassifier() const override;

    };

}
