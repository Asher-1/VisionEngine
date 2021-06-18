#pragma once

#include "../Detector.h"

namespace mirror {
class RetinaFace : public Detector {
public:
	explicit RetinaFace(FaceDetectorType type = FaceDetectorType::RETINA_FACE);
	~RetinaFace() override = default;

protected:
	int loadModel(const char* root_path) override;
	int detectFace(const cv::Mat& img_src, std::vector<FaceInfo>& faces) const override;

private:
	const int RPNs_[3] = { 32, 16, 8 };
	const cv::Size inputSize_ = { 300, 300 };
    std::vector<ANCHORS> anchors_generated_;
};

}