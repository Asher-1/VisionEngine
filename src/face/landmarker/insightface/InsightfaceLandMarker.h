#pragma once

#include "../LandMarker.h"

namespace mirror {
    class InsightfaceLandMarker : public LandMarker {
    public:
        explicit InsightfaceLandMarker(FaceLandMarkerType type = FaceLandMarkerType::INSIGHTFACE_LANDMARKER);

        ~InsightfaceLandMarker() override = default;

    protected:
        int loadModel(const char *root_path) override;

#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int extractKeypoints(const cv::Mat &img_src,
                             const cv::Rect &face, std::vector<cv::Point2f> &keypoints) const override;

    private:
        const float enlarge_scale = 1.5f;
    };

}

