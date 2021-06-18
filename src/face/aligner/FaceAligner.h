#pragma once

#include "opencv2/core.hpp"

namespace mirror {
class FaceAligner {
public:
	FaceAligner();
	~FaceAligner();

	int alignFace(const cv::Mat & img_src,
                  const std::vector<cv::Point2f>& keypoints, cv::Mat& face_aligned) const;

private:
	class Impl;
	Impl* impl_;
};

}