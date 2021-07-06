#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef POSE_EXPORTS
        #define POSE_API __declspec(dllexport)
    #else
        #define POSE_API __declspec(dllimport)
    #endif
#else
    #define POSE_API __attribute__ ((visibility("default")))
#endif

namespace mirror {

class PoseEngine {
public:
	POSE_API ~PoseEngine();
    POSE_API static PoseEngine* GetInstancePtr();
	POSE_API static PoseEngine& GetInstance();
	POSE_API static void ReleaseInstance();
	POSE_API void destroyEngine();

	POSE_API int loadModel(const PoseEngineParams &params);
	POSE_API int updateModel(const PoseEngineParams &params);
	POSE_API int detect(const cv::Mat& img_src, std::vector<PoseResult>& poses) const;
	POSE_API const std::vector<std::pair<int, int>>& getJointPairs() const;

private:
    //! Default constructor
	/** Shouldn't be called directly. Use 'GetUniqueInstance' instead.
	**/
    POSE_API explicit PoseEngine();

private:
	class Impl;
	Impl* impl_;
};

}