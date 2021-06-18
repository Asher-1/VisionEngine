#pragma once

#include <vector>
#include "../common/common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#ifdef CLASSIFIER_EXPORTS
#define CLASSIFIER_API __declspec(dllexport)
#else
#define CLASSIFIER_API __declspec(dllimport)
#endif
#else
#define CLASSIFIER_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
    class ClassifierEngine {
    public:
        CLASSIFIER_API ~ClassifierEngine();

        CLASSIFIER_API static ClassifierEngine *GetInstancePtr();

        CLASSIFIER_API static ClassifierEngine &GetInstance();

        CLASSIFIER_API static void ReleaseInstance();

        CLASSIFIER_API void destroyEngine();

        CLASSIFIER_API int loadModel(const char *root_path, const ClassifierEigenParams &params);

        CLASSIFIER_API int classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const;

    private:
        //! Default constructor
        /** Shouldn't be called directly. Use 'GetUniqueInstance' instead.
        **/
        CLASSIFIER_API explicit ClassifierEngine();

    private:
        class Impl;

        Impl *impl_;

    };
}