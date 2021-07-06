#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef SEGMENT_EXPORTS
        #define SEGMENT_API __declspec(dllexport)
    #else
        #define SEGMENT_API __declspec(dllimport)
    #endif
#else
    #define SEGMENT_API __attribute__ ((visibility("default")))
#endif

namespace mirror {

class SegmentEngine {
public:
	SEGMENT_API ~SegmentEngine();
    SEGMENT_API static SegmentEngine* GetInstancePtr();
	SEGMENT_API static SegmentEngine& GetInstance();
	SEGMENT_API static void ReleaseInstance();
	SEGMENT_API void destroyEngine();

	SEGMENT_API int loadModel(const SegmentEngineParams &params);
	SEGMENT_API int updateModel(const SegmentEngineParams &params);
	SEGMENT_API int detect(const cv::Mat& img_src, std::vector<SegmentInfo>& segments) const;

private:
    //! Default constructor
	/** Shouldn't be called directly. Use 'GetUniqueInstance' instead.
	**/
    SEGMENT_API explicit SegmentEngine();

private:
	class Impl;
	Impl* impl_;
};

}