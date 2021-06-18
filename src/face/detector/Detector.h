#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {
    using ANCHORS = std::vector<cv::Rect>;

    class Detector {
    public:
        explicit Detector(FaceDetectorType type);

        virtual ~Detector();

    public:
        int load(const char *root_path, const FaceEigenParams& params);

        int detect(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const;

        inline FaceDetectorType getType() const { return type_; }

    protected:
        virtual int loadModel(const char *root_path) = 0;

        virtual int detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const = 0;

    protected:
        FaceDetectorType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        bool has_kps_ = true;
        float iouThreshold_ = 0.45f;
        float scoreThreshold_ = 0.5f;
    };

// 工厂基类
    class DetectorFactory {
    public:
        virtual ~DetectorFactory() = default;
        virtual Detector *CreateDetector() const = 0;

    };

// 不同人脸检测器
    class CenterfaceFactory : public DetectorFactory {
    public:
        CenterfaceFactory() = default;
        ~CenterfaceFactory() override = default;

        Detector *CreateDetector() const override;
    };

    class MtcnnFactory : public DetectorFactory {
    public:
        MtcnnFactory() = default;
        ~MtcnnFactory() override = default;

        Detector *CreateDetector() const override;

    };

    class RetinafaceFactory : public DetectorFactory {
    public:
        RetinafaceFactory() = default;

        ~RetinafaceFactory() override = default;

        Detector *CreateDetector() const override;
    };

    class ScrfdFactory : public DetectorFactory {
    public:
        ScrfdFactory() = default;

        ~ScrfdFactory() override = default;

        Detector *CreateDetector() const override;
    };

    class AnticovFactory : public DetectorFactory {
    public:
        AnticovFactory() = default;

        ~AnticovFactory() override = default;

        Detector *CreateDetector() const override;
    };

}