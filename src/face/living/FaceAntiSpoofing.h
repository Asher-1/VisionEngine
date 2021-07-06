#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {

    struct ModelConfig {
        float scale;
        float shift_x;
        float shift_y;
        int height;
        int width;
        std::string name;
        bool org_resize;
    };

    class FaceAntiSpoofing {
    public:
        explicit FaceAntiSpoofing(FaceAntiSpoofingType type);

        virtual ~FaceAntiSpoofing();

        int load(const FaceEngineParams &params);
        int update(const FaceEngineParams &params);

        bool detect(const cv::Mat &src, const cv::Rect &box, float &livingScore) const;

        static cv::Rect CalculateBox(const cv::Rect &box, int w, int h, const ModelConfig &config);

        inline FaceAntiSpoofingType getType() const { return type_; }

    protected:


#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
#endif

        virtual int loadModel(const char *root_path) = 0;

        virtual float detectLiving(const cv::Mat &src, const cv::Rect &box) const = 0;

    private:
        void clearNets();

    protected:
        FaceAntiSpoofingType type_;
        std::vector<ncnn::Net *> nets_;
        std::vector<ModelConfig> configs_;
        int model_num_ = 1;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        float faceLivingThreshold_ = 0.93;
        cv::Size inputSize_ = {80, 80};
        std::string modelPath_;
    };

    class FaceAntiSpoofingFactory {
    public:
        virtual FaceAntiSpoofing *CreateFaceAntiSpoofing() const = 0;

        virtual ~FaceAntiSpoofingFactory() = default;

    };

    class LiveDetectorFactory : public FaceAntiSpoofingFactory {
    public:
        LiveDetectorFactory() = default;

        FaceAntiSpoofing *CreateFaceAntiSpoofing() const override;

        ~LiveDetectorFactory() override = default;
    };

}