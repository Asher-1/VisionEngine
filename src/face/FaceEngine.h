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
        FACE_API ~FaceEngine();

        //! Returns the static and unique instance pointer
        FACE_API static FaceEngine *GetInstancePtr();

        //! Returns the static and unique instance reference
        FACE_API static FaceEngine &GetInstance();

        //! Release the FaceEngine instance
        FACE_API static void ReleaseInstance();

        //! Release the FaceEngine instance
        FACE_API void destroyEngine();

        //! load model from 'FaceEngineParams' params
        FACE_API int loadModel(const FaceEngineParams &params);

        /// \brief Register face
        /// \param imgSrc [in] The input cv::Mat origin image.
        /// \param name [in] The face id or name
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int registerFace(const cv::Mat &imgSrc, const std::string &name);

        /// \brief Verify face
        /// \param imgSrc [in] The input cv::Mat origin image.
        /// \param result [out] The verification result.
        /// \param livingEnabled [in] If true, using living detection.
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int verifyFace(const cv::Mat &imgSrc, VerificationResult &result,
                                bool livingEnabled = false) const;

        /// \brief Verify face
        /// \param imgSrc [in] The input cv::Mat origin image.
        /// \param keyPoints [in] The single cv::Rect detected face box
        /// \param result [out] The verification result.
       /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int verifyFace(const cv::Mat &imgSrc,
                                const std::vector<cv::Point2f> &keyPoints,
                                VerificationResult &result) const;

        /// \brief Detect living face
        /// \param imgSrc [in] The input cv::Mat origin image.
        /// \param livingScore [out] If greater than this score threshold, real face. Otherwise, fake face.
        /// \return Return true if real face else false [please reference to "common.h"].
        FACE_API bool detectLivingFace(const cv::Mat &imgSrc, float &livingScore) const;

        /// \brief Detect living faces
        /// \param imgSrc [in] The input cv::Mat origin image.
        /// \param box [in] The single cv::Rect detected face box
        /// \param livingScore [out] If greater than this score threshold, real face. Otherwise, fake face.
        /// \return Return true if real face else false [please reference to "common.h"].
        FACE_API bool detectLivingFace(const cv::Mat &imgSrc, const cv::Rect &box, float &livingScore) const;

        /// \brief Detect face
        /// \param imgSrc [in] The input cv::Mat image.
        /// \param faces [out] The detected faces information
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int detectFace(const cv::Mat &imgSrc, std::vector<FaceInfo> &faces) const;

        /// \brief Track faces
        /// \param currFaces [in] The current detected faces information.
        /// \param faces [out] The faces will be tracked
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int track(const std::vector<FaceInfo> &currFaces,
                           std::vector<TrackedFaceInfo> &faces);

        /// \brief Detect living faces
        /// \param imgSrc [in] The input cv::Mat image.
        /// \param box [in] The single cv::Rect detected face box
        /// \param keypoints [out] The extracted face keypoints with 2D coordinate like (x, y) * n.
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int extractKeypoints(const cv::Mat &imgSrc,
                                      const cv::Rect &box, std::vector<cv::Point2f> &keypoints) const;

        /// \brief Extract face feature from input aligned image with 112*112
        /// \param imgSrc [in] The input aligned image with 112*112 in cv::Mat format.
        /// \param feature [out] The extracted face feature with kFaceFeatureDim
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int extractFeature(const cv::Mat &imgSrc, std::vector<float> &feature) const;

        /// \brief Extract face feature from input aligned image with 112*112
        /// \param imgSrc [in] The input aligned image with 112*112 in cv::Mat format.
        /// \param keypoints [in] The extracted face keypoints with 2D coordinate like (x, y) * n.
        /// \param faceAligned [out] The aligned image with 112*112 in cv::Mat format.
        /// \return Return 0 if success else ErrorCode [please reference to "common.h"].
        FACE_API int alignFace(const cv::Mat &imgSrc, const std::vector<cv::Point2f> &keypoints,
                               cv::Mat &faceAligned) const;

        //! Load registered faces from database file
        FACE_API int Load();
        //! Save face database cache
        FACE_API int Save() const;
        //! Reset face database
        FACE_API int Clear();
        //! Delete registered face information from database by given face name
        FACE_API int Delete(const std::string &name);
        //! Find all registered faces names from database
        FACE_API int Find(std::vector<std::string> &names) const;

        /// \brief Insert face features into database
        /// \param feat [in] The extracted face feature with kFaceFeatureDim.
        /// \param name [in] The face id or name
        /// \return The new face index if success else ErrorCode [please reference to "common.h"].
        FACE_API int Insert(const std::vector<float> &feat, const std::string &name);

        /// \brief Query the most similarity face from registered faces
        /// \param feat [in] The extracted face feature with kFaceFeatureDim.
        /// \param queryResult [out] The query result with similarity and registered face name
        /// \return The new face index if success else ErrorCode [please reference to "common.h"].
        FACE_API int QueryTop(const std::vector<float> &feat, QueryResult &queryResult) const;

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
