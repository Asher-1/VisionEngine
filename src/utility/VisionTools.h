#pragma once

#include <vector>
#include "../common/common.h"

namespace utility {
    int DrawClassifications(cv::Mat &img_src, const std::vector<mirror::ImageInfo> &images,
                            double fontScale = 0.5, int thickness = 1,
                            const cv::Scalar &fontColor = cv::Scalar(255, 100, 0),
                            int lineType = cv::LINE_8);

    int DrawFaces(cv::Mat &img_src, const std::vector<mirror::FaceInfo> &faces,
                  bool show_kps = false, double fontScale = 0.5, int thickness = 1,
                  const cv::Scalar &boxColor = cv::Scalar(0, 255, 0),
                  const cv::Scalar &fontColor = cv::Scalar(0, 0, 0),
                  const cv::Scalar &bkColor = cv::Scalar(255, 255, 255));

    int DrawObjects(cv::Mat &img_src, const std::vector<mirror::ObjectInfo> &objects,
                    double fontScale = 0.5, int thickness = 1,
                    const cv::Scalar &boxColor = cv::Scalar(0, 255, 0),
                    const cv::Scalar &fontColor = cv::Scalar(0, 0, 0),
                    const cv::Scalar &bkColor = cv::Scalar(255, 255, 255));

    int DrawText(cv::Mat &img_src, const cv::Point2i &position, const char *text,
                 double fontScale = 0.5, int thickness = 1,
                 const cv::Scalar &fontColor = cv::Scalar(0, 0, 0),
                 const cv::Scalar &bkColor = cv::Scalar(255, 255, 255));

    int DrawKeyPoints(cv::Mat &img_src, const std::vector<cv::Point2f> &keyPoints,
                      int radius = 2, const cv::Scalar &color = cv::Scalar(0, 255, 255),
                      int thickness = -1);
}