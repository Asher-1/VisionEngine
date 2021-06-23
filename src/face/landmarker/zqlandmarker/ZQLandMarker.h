#pragma once

#include "../LandMarker.h"

namespace mirror {
    class ZQLandMarker : public LandMarker {
    public:
        explicit ZQLandMarker(FaceLandMarkerType type = FaceLandMarkerType::ZQ_LANDMARKER);

        ~ZQLandMarker() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int extractKeypoints(const cv::Mat &img_src,
                             const cv::Rect &face, std::vector<cv::Point2f> &keypoints) const override;

    private:
        const float meanVals[3] = {127.5f, 127.5f, 127.5f};
        const float normVals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    };

}

