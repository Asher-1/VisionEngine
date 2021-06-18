#pragma once

#include "../Detector.h"
#include <vector>
#include "opencv2/core.hpp"

namespace mirror {
class CenterFace : public Detector {
public:
    explicit CenterFace(FaceDetectorType type = FaceDetectorType::CENTER_FACE);
    ~CenterFace() override = default;
protected:
	int loadModel(const char* root_path) override;
	int detectFace(const cv::Mat& img_src, std::vector<FaceInfo>& faces) const override;
};

}