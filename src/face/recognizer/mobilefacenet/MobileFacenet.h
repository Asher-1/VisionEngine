#pragma once

#include "../Recognizer.h"

namespace mirror {

class MobileFacenet : public Recognizer {
public:
    explicit MobileFacenet(FaceRecognizerType type = FaceRecognizerType::ARC_FACE);
	~MobileFacenet() override = default;

protected:
	int loadModel(const char* root_path) override;
	int extractFeature(const cv::Mat& img_face, std::vector<float>& feature) const override;
};

}