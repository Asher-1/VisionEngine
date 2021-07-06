#include "VisionTools.h"

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace utility {
    using namespace mirror;

    cv::Scalar GetRandomColor() {
        cv::RNG rng;
        return {rng.uniform(0.0, 256.0),
                rng.uniform(0.0, 256.0),
                rng.uniform(0.0, 256.0)};
    }

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
        } else {
            position.x = -1;
            position.y = -1;
        }

        return 0;
    }

    int DrawPoses(cv::Mat &img_src, const std::vector<mirror::PoseResult> &poses,
                  const std::vector<std::pair<int, int>> &jointPairs,
                  int radius, int thickness, const cv::Scalar &pointColor) {
        // draw bone
        const cv::Point2f absentKeypoint(-1.0f, -1.0f);
        for (const auto &pose : poses) {
            cv::Scalar color = GetRandomColor();
            for (const auto &keyPoint : pose.keyPoints) {
                if (keyPoint.p != absentKeypoint) {
                    cv::circle(img_src, keyPoint.p, radius, pointColor, -1);
                }
            }

            for (const auto &limbKeypointsId : jointPairs) {
                std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keyPoints[limbKeypointsId.first].p,
                                                                  pose.keyPoints[limbKeypointsId.second].p);
                if (limbKeypoints.first == absentKeypoint
                    || limbKeypoints.second == absentKeypoint) {
                    continue;
                }
                cv::line(img_src, limbKeypoints.first, limbKeypoints.second, color, thickness);
            }

            if (pose.boxInfo.location_.area() > 10) {
                cv::rectangle(img_src, pose.boxInfo.location_, color, thickness);
            }
        }

        return 0;
    }

    int DrawMask(cv::Mat &img_src, const std::vector<mirror::SegmentInfo> &segments,
                 int maskType, double fontScale, int thickness, const cv::Scalar &fontColor,
                 const cv::Scalar &bkColor) {
        if (segments.empty()) return 1;
        static const unsigned char colors[19][3] = {
                {128, 64,  128},
                {244, 35,  232},
                {70,  70,  70},
                {102, 102, 156},
                {190, 153, 153},
                {153, 153, 153},
                {250, 170, 30},
                {220, 220, 0},
                {107, 142, 35},
                {152, 251, 152},
                {70,  130, 180},
                {220, 20,  60},
                {255, 0,   0},
                {0,   0,   142},
                {0,   0,   70},
                {0,   60,  100},
                {0,   80,  100},
                {0,   0,   230},
                {119, 11,  32}
        };

        if (maskType == 0) {
            int color_index = 0;
            for (size_t i = 0; i < segments.size(); i++) {
                const SegmentInfo &obj = segments[i];
                const unsigned char *color = colors[color_index++];

                if (obj.boxInfo.score_ < 0.15)
                    continue;

//                fprintf(stderr, "%s = %.5f at %i %i %i x %i\n", obj.boxInfo.name_.c_str(), obj.boxInfo.score_,
//                        obj.boxInfo.location_.x, obj.boxInfo.location_.y,
//                        obj.boxInfo.location_.width, obj.boxInfo.location_.height);

                cv::rectangle(img_src, obj.boxInfo.location_, cv::Scalar(color[0], color[1], color[2]));

                char text[256];
                sprintf(text, "%s %.1f%%", obj.boxInfo.name_.c_str(), obj.boxInfo.score_ * 100);

                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                                      fontScale, thickness, &baseLine);

                int x = obj.boxInfo.location_.x;
                int y = obj.boxInfo.location_.y - label_size.height - baseLine;
                if (y < 0)
                    y = 0;
                if (x + label_size.width > img_src.cols) {
                    x = img_src.cols - label_size.width;
                }

                cv::rectangle(img_src,
                              cv::Rect(cv::Point(x, y),
                                       cv::Size(label_size.width, label_size.height + baseLine)),
                              bkColor, -1);

                cv::putText(img_src, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor);

                // draw mask
                for (int y = 0; y < img_src.rows; y++) {
                    const uchar *mp = obj.mask.ptr(y);
                    uchar *p = img_src.ptr(y);
                    for (int x = 0; x < img_src.cols; x++) {
                        if (mp[x] == 255) {
                            p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                            p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                            p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                        }
                        p += 3;
                    }
                }
            }
        } else {
            const SegmentInfo &obj = segments[0];
            // draw mask
            for (int y = 0; y < img_src.rows; y++) {
                const uchar *mp = obj.mask.ptr(y);
                uchar *p = img_src.ptr(y);
                for (int x = 0; x < img_src.cols; x++) {
                    if (mp[x] >= 19) {
                        continue;
                    }
                    const unsigned char *color = colors[int(mp[x])];
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                    p += 3;
                }
            }
        }

        return 0;
    }
}

