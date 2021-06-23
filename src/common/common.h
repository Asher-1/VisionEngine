#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#endif


namespace mirror {
#define kFaceFeatureDim 128
#define kFaceNameDim 256

    // common
    const int CUSTOM_THREAD_NUMBER = 2;
    namespace ErrorCode {
        const int NULL_ERROR = 10000;
        const int EMPTY_INPUT_ERROR = 10001;
        const int UNINITIALIZED_ERROR = 10002;
        const int MODEL_LOAD_ERROR = 10003;
        const int MODEL_UPDATE_ERROR = 10004;
        const int DIMENSION_MISS_MATCH_ERROR = 10005;
    }


    // for classifier module
    enum ClassifierType {
        MOBILE_NET = 0,
    };

    std::string GetClassifierTypeName(ClassifierType type);

    struct ImageInfo {
        std::string label_;
        float score_;
    };

    struct ClassifierEigenParams {
        int topK = 5;
        bool verbose = false;
        bool thread_num = -1;
        bool gpuEnabled = false;
        std::string model_path;
        ClassifierType classifierType = ClassifierType::MOBILE_NET;
#if defined __ANDROID__
        AAssetManager* mgr = nullptr;
#endif
    };

    // for object module
    enum ObjectDetectorType {
        YOLOV5 = 0,
        MOBILENET_SSD = 1,
    };

    std::string GetObjectDetectorTypeName(ObjectDetectorType type);

    struct ObjectInfo {
        cv::Rect location_;
        float score_;
        std::string name_;
    };

    struct ObjectEigenParams {
        std::string model_path;
        bool gpuEnabled = false;
        bool thread_num = -1;
        bool verbose = false;
        float nmsThreshold = -1.0f;
        float scoreThreshold = -1.0f;
        ObjectDetectorType objectDetectorType = ObjectDetectorType::MOBILENET_SSD;
#if defined __ANDROID__
        AAssetManager* mgr = nullptr;
#endif
    };

    // for face module
    enum FaceAntiSpoofingType {
        LIVE_FACE = 0,
    };

    enum FaceLandMarkerType {
        INSIGHTFACE_LANDMARKER = 0,
        ZQ_LANDMARKER = 1,
    };

    enum FaceDetectorType {
        RETINA_FACE = 0,
        SCRFD_FACE = 1,
        CENTER_FACE = 2,
        MTCNN_FACE = 3,
        ANTICOV_FACE = 4,
    };

    enum FaceRecognizerType {
        ARC_FACE = 0,
    };

    struct FaceInfo {
        cv::Rect location_;
        cv::Point2f keypoints_[5];
        float score_;
        bool mask_;
    };

    struct TrackedFaceInfo {
        FaceInfo face_info_;
        float iou_score_;
    };

    struct QueryResult {
        std::string name_;
        float sim_;
    };

    struct FaceEigenParams {
        std::string modelPath;
        std::string faceFeaturePath;
        bool gpuEnabled = false;
        bool verbose = false;
        bool threadNum = -1;
        float nmsThreshold = -1.0f;
        float scoreThreshold = -1.0f;
        float livingThreshold = -1.0f;
        bool faceDetectorEnabled = true;
        bool faceRecognizerEnabled = true;
        bool faceAntiSpoofingEnabled = false;
        bool faceLandMarkerEnabled = false;
        FaceAntiSpoofingType faceAntiSpoofingType = FaceAntiSpoofingType::LIVE_FACE;
        FaceLandMarkerType faceLandMarkerType = FaceLandMarkerType::INSIGHTFACE_LANDMARKER;
        FaceDetectorType faceDetectorType = FaceDetectorType::RETINA_FACE;
        FaceRecognizerType faceRecognizerType = FaceRecognizerType::ARC_FACE;
#if defined __ANDROID__
        AAssetManager* mgr = nullptr;
#endif
    };

    std::string GetAntiSpoofingTypeName(FaceAntiSpoofingType type);

    std::string GetLandMarkerTypeName(FaceLandMarkerType type);

    std::string GetRecognizerTypeName(FaceRecognizerType type);

    std::string GetDetectorTypeName(FaceDetectorType type);

    int RatioAnchors(const cv::Rect &anchor,
                     const std::vector<float> &ratios, std::vector<cv::Rect> &anchors);

    int ScaleAnchors(const std::vector<cv::Rect> &ratio_anchors,
                     const std::vector<float> &scales, std::vector<cv::Rect> &anchors);

    int GenerateAnchors(const int &base_size,
                        const std::vector<float> &ratios,
                        const std::vector<float> &scales,
                        std::vector<cv::Rect> &anchors);

    float InterRectArea(const cv::Rect &a,
                        const cv::Rect &b);

    int ComputeIOU(const cv::Rect &rect1,
                   const cv::Rect &rect2, float *iou,
                   const std::string &type = "UNION");

    template<typename T>
    int NMS(const std::vector<T> &inputs, std::vector<T> &result,
            const float &threshold, const std::string &type = "UNION") {
        result.clear();
        if (inputs.empty())
            return -1;

        std::vector<T> inputs_tmp;
        inputs_tmp.assign(inputs.begin(), inputs.end());
        std::sort(inputs_tmp.begin(), inputs_tmp.end(),
                  [](const T &a, const T &b) {
                      return a.score_ > b.score_;
                  });

        std::vector<int> indexes(inputs_tmp.size());

        for (int i = 0; i < indexes.size(); i++) {
            indexes[i] = i;
        }

        while (!indexes.empty()) {
            int good_idx = indexes[0];
            result.push_back(inputs_tmp[good_idx]);
            std::vector<int> tmp_indexes = indexes;
            indexes.clear();
            for (int i = 1; i < tmp_indexes.size(); i++) {
                int tmp_i = tmp_indexes[i];
                float iou = 0.0f;
                ComputeIOU(inputs_tmp[good_idx].location_, inputs_tmp[tmp_i].location_, &iou, type);
                if (iou <= threshold) {
                    indexes.push_back(tmp_i);
                }
            }
        }
        return 0;
    }

    float CalculateSimilarity(const std::vector<float> &feature1, const std::vector<float> &feature2);

    void EnlargeRect(const float &scale, cv::Rect &rect);

    void RectifyRect(cv::Rect &rect);

    void ConvertKeyPoints(const cv::Point2f ori[], int size, std::vector<cv::Point2f> &dst);

}