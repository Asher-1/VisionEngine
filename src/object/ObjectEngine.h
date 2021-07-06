#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef OBJECT_EXPORTS
        #define OBJECT_API __declspec(dllexport)
    #else
        #define OBJECT_API __declspec(dllimport)
    #endif
#else
    #define OBJECT_API __attribute__ ((visibility("default")))
#endif

namespace mirror {

class ObjectEngine {
public:
	OBJECT_API ~ObjectEngine();
    OBJECT_API static ObjectEngine* GetInstancePtr();
	OBJECT_API static ObjectEngine& GetInstance();
	OBJECT_API static void ReleaseInstance();
	OBJECT_API void destroyEngine();

	OBJECT_API int loadModel(const ObjectEngineParams &params);
	OBJECT_API int updateModel(const ObjectEngineParams &params);
	OBJECT_API int detect(const cv::Mat& img_src, std::vector<ObjectInfo>& objects) const;

private:
    //! Default constructor
	/** Shouldn't be called directly. Use 'GetUniqueInstance' instead.
	**/
    OBJECT_API explicit ObjectEngine();

private:
	class Impl;
	Impl* impl_;
};

}