#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../common/common.h"

namespace mirror {
class Tracker {
public:
    Tracker() = default;
    ~Tracker() = default;
    int track(const std::vector<FaceInfo>& curr_faces, std::vector<TrackedFaceInfo>& faces);

private:
    std::vector<TrackedFaceInfo> pre_tracked_faces_;
    const float minScore_ = 0.3f;
    const float maxScore_ = 0.5f;
};

}