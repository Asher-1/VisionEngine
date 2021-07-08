#include "common.h"
#include <algorithm>
#include <iostream>

namespace mirror {
    int RatioAnchors(const cv::Rect &anchor,
                     const std::vector<float> &ratios,
                     std::vector<cv::Rect> &anchors) {
        anchors.clear();
        cv::Point center = cv::Point(anchor.x + (anchor.width - 1) * 0.5f,
                                     anchor.y + (anchor.height - 1) * 0.5f);
        float anchor_size = anchor.width * anchor.height;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(CUSTOM_THREAD_NUMBER)
#endif
        for (int i = 0; i < static_cast<int>(ratios.size()); ++i) {
            float ratio = ratios.at(i);
            float anchor_size_ratio = anchor_size / ratio;
            float curr_anchor_width = std::sqrt(anchor_size_ratio);
            float curr_anchor_height = curr_anchor_width * ratio;
            float curr_x = center.x - (curr_anchor_width - 1) * 0.5f;
            float curr_y = center.y - (curr_anchor_height - 1) * 0.5f;

            cv::Rect curr_anchor = cv::Rect(curr_x, curr_y,
                                            curr_anchor_width - 1, curr_anchor_height - 1);
            anchors.push_back(curr_anchor);
        }
        return 0;
    }

    int ScaleAnchors(const std::vector<cv::Rect> &ratio_anchors,
                     const std::vector<float> &scales, std::vector<cv::Rect> &anchors) {
        anchors.clear();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(CUSTOM_THREAD_NUMBER)
#endif
        for (int i = 0; i < static_cast<int>(ratio_anchors.size()); ++i) {
            cv::Rect anchor = ratio_anchors.at(i);
            cv::Point2f center = cv::Point2f(anchor.x + anchor.width * 0.5f,
                                             anchor.y + anchor.height * 0.5f);
            for (int j = 0; j < static_cast<int>(scales.size()); ++j) {
                float scale = scales.at(j);
                float curr_width = scale * (anchor.width + 1);
                float curr_height = scale * (anchor.height + 1);
                float curr_x = center.x - curr_width * 0.5f;
                float curr_y = center.y - curr_height * 0.5f;
                cv::Rect curr_anchor = cv::Rect(curr_x, curr_y,
                                                curr_width, curr_height);
                anchors.push_back(curr_anchor);
            }
        }

        return 0;
    }

    int GenerateAnchors(const int &base_size,
                        const std::vector<float> &ratios,
                        const std::vector<float> &scales,
                        std::vector<cv::Rect> &anchors) {
        anchors.clear();
        cv::Rect anchor = cv::Rect(0, 0, base_size, base_size);
        std::vector<cv::Rect> ratio_anchors;
        RatioAnchors(anchor, ratios, ratio_anchors);
        ScaleAnchors(ratio_anchors, scales, anchors);

        return 0;
    }

    float InterRectArea(const cv::Rect &a, const cv::Rect &b) {
        cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
        cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
        cv::Point diff = right_bottom - left_top;
        return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
    }

    int ComputeIOU(const cv::Rect &rect1,
                   const cv::Rect &rect2, float *iou,
                   const std::string &type) {

        float inter_area = InterRectArea(rect1, rect2);
        if (type == "UNION") {
            *iou = inter_area / (rect1.area() + rect2.area() - inter_area);
        } else {
            *iou = inter_area / MIN(rect1.area(), rect2.area());
        }

        return 0;
    }

    float CalculateSimilarity(const std::vector<float> &feature1, const std::vector<float> &feature2) {
        if (feature1.size() != feature2.size()) {
            std::cout << "feature size not match." << std::endl;
            return 10003;
        }
        float inner_product = 0.0f;
        float feature_norm1 = 0.0f;
        float feature_norm2 = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(CUSTOM_THREAD_NUMBER)
#endif
        for (int i = 0; i < kFaceFeatureDim; ++i) {
            inner_product += feature1[i] * feature2[i];
            feature_norm1 += feature1[i] * feature1[i];
            feature_norm2 += feature2[i] * feature2[i];
        }
        return inner_product / sqrt(feature_norm1) / sqrt(feature_norm2);
    }

    void EnlargeRect(const float &scale, cv::Rect &rect) {
        float offset_x = (scale - 1.f) / 2.f * rect.width;
        float offset_y = (scale - 1.f) / 2.f * rect.height;
        rect.x -= offset_x;
        rect.y -= offset_y;
        rect.width = scale * rect.width;
        rect.height = scale * rect.height;
    }

    void RectifyRect(cv::Rect &rect) {
        int max_side = MAX(rect.width, rect.height);
        int offset_x = (max_side - rect.width) / 2;
        int offset_y = (max_side - rect.height) / 2;

        rect.x -= offset_x;
        rect.y -= offset_y;
        rect.width = max_side;
        rect.height = max_side;
    }

    std::string GetClassifierTypeName(ClassifierType type) {
        switch (type) {
            case MOBILE_NET:
                return "MOBILE_NET";
            case SQUEEZE_NET:
                return "SQUEEZE_NET";
            default:
                return "SQUEEZE_NET";
        }
    }

    std::string GetLandMarkerTypeName(FaceLandMarkerType type) {
        switch (type) {
            case INSIGHTFACE_LANDMARKER:
                return "INSIGHTFACE_LANDMARKER";
            case ZQ_LANDMARKER:
                return "ZQ_LANDMARKER";
            default:
                return "NONE";
        }
    }

    std::string GetRecognizerTypeName(FaceRecognizerType type) {
        switch (type) {
            case ARC_FACE:
                return "ARC_FACE";
            default:
                return "NONE";
        }
    }

    std::string GetDetectorTypeName(FaceDetectorType type) {
        switch (type) {
            case ANTICOV_FACE:
                return "ANTICOV_FACE";
            case CENTER_FACE:
                return "CENTER_FACE";
            case MTCNN_FACE:
                return "MTCNN_FACE";
            case RETINA_FACE:
                return "RETINA_FACE";
            case SCRFD_FACE:
                return "SCRFD_FACE";
            default:
                return "NONE";
        }
    }

    std::string GetAntiSpoofingTypeName(FaceAntiSpoofingType type) {
        switch (type) {
            case LIVE_FACE:
                return "LIVE_FACE";
            default:
                return "NONE";
        }
    }

    std::string GetObjectDetectorTypeName(ObjectDetectorType type) {
        switch (type) {
            case YOLOV4:
                return "YOLOV4";
            case YOLOV5:
                return "YOLOV5";
            case NANO_DET:
                return "NANO_DET";
            case MOBILENET_SSD:
                return "MOBILENET_SSD";
            default:
                return "YOLOV4";
        }
    }

    std::string GetPoseEstimationTypeName(PoseEstimationType type) {
        switch (type) {
            case SIMPLE_POSE:
                return "SIMPLE_POSE";
            case LIGHT_OPEN_POSE:
                return "LIGHT_OPEN_POSE";
            default:
                return "SIMPLE_POSE";
        }
    }

    std::string GetSegmentTypeName(SegmentType type) {
        switch (type) {
            case YOLACT_SEG:
                return "YOLACT_SEG";
            case MOBILENETV3_SEG:
                return "MOBILENETV3_SEG";
        }
    }

    void ConvertKeyPoints(const cv::Point2f ori[], int size, std::vector<cv::Point2f> &dst) {
        dst = std::vector<cv::Point2f>(ori, ori + size);
    }

    void ComputeMeanAndVariance2(const std::vector<float> &inputs, double &mean, double &variance) {
        double _mean = 0.0, _std2 = 0.0;
        std::size_t count = 0;

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            _mean += inputs[i];
            _std2 += static_cast<double>(inputs[i]) * inputs[i];
            ++count;
        }

        if (count) {
            _mean /= count;
            mean = static_cast<double>(_mean);
            _std2 = std::abs(_std2 / count - _mean * _mean);
            variance = static_cast<double>(_std2);
        } else {
            mean = 0;
            variance = 0;
        }
    }

    void SplitString(const std::string &str, const std::string &delimiter,
                     int offset, std::vector<std::string> &result) {
        result.clear();
        std::string::size_type pos = 0;
        std::string::size_type prev = 0;
        while ((pos = str.find(delimiter, prev)) != std::string::npos) {
            prev += offset;
            result.push_back(str.substr(prev, pos - prev));
            prev = pos + 1;
        }

        // To get the last substring (or only, if delimiter is not found)
        result.push_back(str.substr(prev + 10));
    }

}
