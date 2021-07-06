#pragma once

#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {
    class SegmentDetector {
    public:
        using Super = SegmentDetector;

        explicit SegmentDetector(SegmentType type);

        virtual ~SegmentDetector();

        int load(const SegmentEngineParams &params);

        int update(const SegmentEngineParams &params);

        int detect(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const;

        inline SegmentType getType() const { return type_; }

    protected:

        int loadModel(const char *params, const char *models);

#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
        int loadModel(AAssetManager* mgr, const char* params, const char* models);
#endif

        virtual int loadModel(const char *root_path) = 0;

        virtual int detectSeg(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const = 0;

    protected:
        SegmentType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        float scoreThreshold_ = 0.7f;
        float nmsThreshold_ = 0.5f;
        std::vector<std::string> class_names_;
        cv::Size inputSize_ = {640, 640};
        std::string modelPath_;
        float meanVals[3] = {123.68f, 116.28f, 103.53f};
        float normVals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    };

    class SegmentDetectorFactory {
    public:
        virtual SegmentDetector *createDetector() const = 0;

        virtual ~SegmentDetectorFactory() = default;
    };

    class YolactSegFactory : public SegmentDetectorFactory {
    public:
        YolactSegFactory() = default;

        SegmentDetector *createDetector() const override;

        ~YolactSegFactory() override = default;
    };

    class MobileNetV3SegFactory : public SegmentDetectorFactory {
    public:
        MobileNetV3SegFactory() = default;

        SegmentDetector *createDetector() const override;

        ~MobileNetV3SegFactory() override = default;
    };

}