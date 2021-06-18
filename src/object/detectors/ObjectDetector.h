#pragma once

#include <vector>
#include "opencv2/core.hpp"
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {
    class ObjectDetector {
    public:
        explicit ObjectDetector(ObjectDetectorType type);

        virtual ~ObjectDetector();

        int load(const char *root_path, const ObjectEigenParams &params);

        int detect(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const;

        inline ObjectDetectorType getType() const { return type_; }

    protected:
        virtual int loadModel(const char *root_path) = 0;

        virtual int detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const = 0;

    protected:
        ObjectDetectorType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        float scoreThreshold_ = 0.7f;
        float nmsThreshold_ = 0.5f;
        std::vector<std::string> class_names_;
        cv::Size inputSize_ = {640, 640};
    };

    class ObjectDetectorFactory {
    public:
        virtual ObjectDetector *createDetector() const = 0;

        virtual ~ObjectDetectorFactory() = default;
    };

    class MobilenetSSDFactory : public ObjectDetectorFactory {
    public:
        MobilenetSSDFactory() = default;

        ObjectDetector *createDetector() const override;

        ~MobilenetSSDFactory() override = default;
    };

    class Yolov5Factory : public ObjectDetectorFactory {
    public:
        Yolov5Factory() = default;

        ObjectDetector *createDetector() const override;

        ~Yolov5Factory() override = default;
    };

}