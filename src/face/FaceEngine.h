#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#ifdef FACE_EXPORTS
#define FACE_API __declspec(dllexport)
#else
#define FACE_API __declspec(dllimport)
#endif
#else
#define FACE_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
    class FaceEngine {
    public:
        //! Returns the static and unique instance
        FACE_API static FaceEngine *GetInstancePtr();

        FACE_API static FaceEngine &GetInstance();

        FACE_API static void ReleaseInstance();

        FACE_API ~FaceEngine();

        FACE_API void destroyEngine();

        FACE_API int loadModel(const FaceEigenParams &params);

        FACE_API bool detectLivingFace(const cv::Mat &img_src, const cv::Rect &box, float &livingScore) const;

        FACE_API int detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const;

        FACE_API int track(const std::vector<FaceInfo> &curr_faces,
                           std::vector<TrackedFaceInfo> &faces);

        FACE_API int extractKeypoints(const cv::Mat &img_src,
                                      const cv::Rect &face, std::vector<cv::Point2f> &keypoints) const;

        FACE_API int extractFeature(const cv::Mat &img_face, std::vector<float> &feature) const;

        FACE_API int alignFace(const cv::Mat &img_src, const std::vector<cv::Point2f> &keypoints,
                               cv::Mat &face_aligned) const;

        // database operation
        FACE_API int Load();

        FACE_API int Save() const;

        FACE_API int Clear();

        FACE_API int Delete(const std::string &name);

        FACE_API int Insert(const std::vector<float> &feat, const std::string &name);

        FACE_API int64_t QueryTop(const std::vector<float> &feat, QueryResult &query_result) const;

    private:
        //! Default constructor
        /** Shouldn't be called directly. Use 'GetUniqueInstance' instead.
        **/
        FACE_API explicit FaceEngine();

    private:
        class Impl;

        Impl *impl_;

    };

}
