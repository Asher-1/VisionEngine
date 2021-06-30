#pragma once

#include "../PoseDetector.h"

template<typename T, std::size_t N>
constexpr std::size_t arraySize(const T (&)[N]) noexcept {
    return N;
}

namespace human_pose_estimation {
    struct Peak {
        Peak(const int id = -1,
             const cv::Point2f &pos = cv::Point2f(),
             const float score = 0.0f);

        int id;
        cv::Point2f pos;
        float score;
    };

    struct HumanPoseByPeaksIndices {
        explicit HumanPoseByPeaksIndices(const int keypointsNumber);

        std::vector<int> peaksIndices;
        int nJoints;
        float score;
    };

    struct TwoJointsConnection {
        TwoJointsConnection(const int firstJointIdx,
                            const int secondJointIdx,
                            const float score);

        int firstJointIdx;
        int secondJointIdx;
        float score;
    };

    void findPeaks(const std::vector<cv::Mat> &heatMaps,
                   float minPeaksDistance,
                   std::vector<std::vector<Peak> > &allPeaks,
                   int heatMapId);

    int groupPeaksToPoses(
            const std::vector<std::vector<Peak> > &allPeaks,
            const std::vector<cv::Mat> &pafs,
            const size_t keypointsNumber,
            const float midPointsScoreThreshold,
            const float foundMidPointsRatioThreshold,
            const int minJointsNumber,
            const float minSubsetScore,
            std::vector<mirror::PoseResult> &poses);
} // namespace human_pose_estimation


namespace mirror {
    class LightOpenPose : public PoseDetector {
    public:
        explicit LightOpenPose(PoseEstimationType type = PoseEstimationType::LIGHT_OPEN_POSE);

        ~LightOpenPose() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectPose(const cv::Mat &img_src, std::vector<PoseResult> &poses) const override;

    private:
        const float meanVals[3] = {127.5f, 127.5f, 127.5f};
        const float normVals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    };

}