#pragma once

#include <opencv2/core.hpp>
#include "../common/common.h"

namespace ncnn {
    class Net;
};

namespace mirror {
    class LandMarker {
    public:
        using Super = LandMarker;
        explicit LandMarker(FaceLandMarkerType type);

        virtual ~LandMarker();

        int load(const FaceEngineParams &params);
        int update(const FaceEngineParams &params);

        int extract(const cv::Mat &img_src, const cv::Rect &face, std::vector<cv::Point2f> &keypoints) const;

        inline FaceLandMarkerType getType() const { return type_; }

    protected:

#if defined __ANDROID__
        virtual int loadModel(AAssetManager* mgr) { return -1; };
        int loadModel(AAssetManager* mgr, const char* params, const char* models);
#endif

        int loadModel(const char *params, const char *models);

        virtual int loadModel(const char *root_path) = 0;

        virtual int extractKeypoints(const cv::Mat &img_src, const cv::Rect &face,
                                     std::vector<cv::Point2f> &keypoints) const = 0;

    protected:
        FaceLandMarkerType type_;
        ncnn::Net *net_ = nullptr;
        bool verbose_ = false;
        bool gpu_mode_ = false;
        bool initialized_ = false;
        cv::Size inputSize_ = {112, 112};
        std::string modelPath_;
    };

    class LandmarkerFactory {
    public:
        virtual LandMarker *CreateLandmarker() const = 0;

        virtual ~LandmarkerFactory() = default;
    };

    class ZQLandMarkerFactory : public LandmarkerFactory {
    public:
        ZQLandMarkerFactory() = default;

        LandMarker *CreateLandmarker() const override;

        ~ZQLandMarkerFactory() override = default;
    };

    class InsightfaceLandMarkerFactory : public LandmarkerFactory {
    public:
        InsightfaceLandMarkerFactory() = default;

        LandMarker *CreateLandmarker() const override;

        ~InsightfaceLandMarkerFactory() override = default;
    };

}

