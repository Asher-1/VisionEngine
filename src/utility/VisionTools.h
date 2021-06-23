#pragma once

#include <vector>
#include "common.h"

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

    /**
     * @param img_src Input image mat.
     * @param text Input text string.
     * @param orientation text corner position optional items {0: top left, 1: top right, 2, bottom left, 3, bottom right}.
     * @param fontScale Font scale factor that is multiplied by the font-specific base size.
     * @param thickness Thickness of lines used to render the text. See #putText for details.
     * @param[out] The text corner position.
     * @return The status flag.
     *
     * */
    int GetTextCornerPosition(const cv::Mat &img_src, const char *text, int orientation,
                              double fontScale, int thickness, cv::Point2i& position);
}