#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {
    class Recognizer {
    public:
        explicit Recognizer(FaceRecognizerType type);

        virtual ~Recognizer();

        int load(const FaceEigenParams &params);
        int update(const FaceEigenParams &params);

        int extract(const cv::Mat &img_face, std::vector<float> &feature) const;

        inline FaceRecognizerType getType() const { return type_; }

    protected:

#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
        int loadModel(AAssetManager* mgr, const char* params, const char* models);
#endif
        virtual int loadModel(const char *root_path) = 0;

        int loadModel(const char *params, const char *models);

        virtual int extractFeature(const cv::Mat &img_face, std::vector<float> &feature) const = 0;

    protected:
        FaceRecognizerType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        int faceFaceFeatureDim_ = kFaceFeatureDim;
        cv::Size inputSize_ = {112, 112};
        std::string modelPath_;

    };

    class RecognizerFactory {
    public:
        virtual ~RecognizerFactory() = default;

        virtual Recognizer *CreateRecognizer() const = 0;
    };

    class MobilefacenetRecognizerFactory : public RecognizerFactory {
    public:
        MobilefacenetRecognizerFactory() = default;

        ~MobilefacenetRecognizerFactory() override = default;

        Recognizer *CreateRecognizer() const override;

    };

}