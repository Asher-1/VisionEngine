#include "VisionTools.h"

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace utility {
    using namespace mirror;

    int DrawClassifications(cv::Mat &img_src, const std::vector<ImageInfo> &images,
                            double fontScale, int thickness, const cv::Scalar &fontColor, int lineType) {
        std::size_t topk = images.size();
        for (std::size_t i = 0; i < topk; ++i) {
            char text[256];
            sprintf(text, "%s %.1f%%", images[i].label_.c_str(), images[i].score_ * 100);
            cv::putText(img_src, text, cv::Point(10, 10 + 30 * i),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, thickness, lineType);
        }
        return 0;
    }

    int DrawFaces(cv::Mat &img_src, const std::vector<FaceInfo> &faces, bool show_kps,
                  double fontScale, int thickness, const cv::Scalar &boxColor,
                  const cv::Scalar &fontColor, const cv::Scalar &bkColor) {
        for (const FaceInfo &face: faces) {
            cv::rectangle(img_src, face.location_, boxColor);

            if (show_kps) {
                std::vector<cv::Point2f> keyPoints;
                ConvertKeyPoints(face.keypoints_, 5, keyPoints);
                DrawKeyPoints(img_src, keyPoints, 2, boxColor);
            }

            char text[256];
            sprintf(text, "%.1f%%", face.score_ * 100);
            DrawText(img_src, face.location_.tl(), text, fontScale, thickness, fontColor, bkColor);
        }

        return 0;
    }

    int DrawObjects(cv::Mat &img_src, const std::vector<ObjectInfo> &objects,
                    double fontScale, int thickness, const cv::Scalar &boxColor,
                    const cv::Scalar &fontColor, const cv::Scalar &bkColor) {
        for (const auto &obj : objects) {
            cv::rectangle(img_src, obj.location_, boxColor, thickness);

            char text[256];
            sprintf(text, "%s %.1f%%", obj.name_.c_str(), obj.score_ * 100);
            DrawText(img_src, obj.location_.tl(), text, fontScale, thickness, fontColor, bkColor);
        }
        return 0;
    }

    int DrawText(cv::Mat &img_src, const cv::Point2i &position, const char *text,
                 double fontScale, int thickness, const cv::Scalar &fontColor,
                 const cv::Scalar &bkColor) {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                              fontScale, thickness, &baseLine);

        int x = position.x;
        int y = position.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > img_src.cols)
            x = img_src.cols - label_size.width;

        cv::rectangle(img_src,
                      cv::Rect(cv::Point(x, y),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                      bkColor, -1);

        cv::putText(img_src, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, thickness);
        return std::max(y - 1, 0);
    }

    int DrawKeyPoints(cv::Mat &img_src, const std::vector<cv::Point2f> &keyPoints,
                      int radius, const cv::Scalar &color, int thickness) {
        for (const auto &keypoint : keyPoints) {
            cv::circle(img_src, keypoint, radius, color, thickness);
        }
        return 0;
    }

    int GetTextCornerPosition(const cv::Mat &img_src, const char *text, int orientation,
                              double fontScale, int thickness, cv::Point2i &position) {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                              fontScale, thickness, &baseLine);
        if (orientation == 0) { // top left
            position.x = 0;
            position.y = label_size.height + 1;
        } else if (orientation == 1) { // top right
            position.x = std::max(img_src.cols - label_size.width - 1, 0);
            position.y = label_size.height + 1;
        } else if (orientation == 2) { //  bottom left
            position.x = 0;
            position.y = std::max(img_src.rows - 1, 0);
        } else if (orientation == 3) { // bottom right
            position.x = std::max(img_src.cols - label_size.width - 1, 0);
            position.y = std::max(img_src.rows - 1, 0);
        }
        else {
            position.x = -1;
            position.y = -1;
        }

        return 0;
    }

}

