#pragma once

#include <vector>
#include "common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#ifdef OCR_EXPORTS
#define OCR_API __declspec(dllexport)
#else
#define OCR_API __declspec(dllimport)
#endif
#else
#define OCR_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
    class OcrEngine {
    public:
        OCR_API ~OcrEngine();

        OCR_API static OcrEngine *GetInstancePtr();

        OCR_API static OcrEngine &GetInstance();

        OCR_API static void ReleaseInstance();

        OCR_API void destroyEngine();

        OCR_API int loadModel(const OcrEngineParams &params);

        OCR_API int updateModel(const OcrEngineParams &params);

        OCR_API int detectText(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const;

        OCR_API int recognizeText(const cv::Mat &img_src,
                                         const std::vector<TextBox> &textBoxes,
                                         std::vector<OCRResult> &ocrResults) const;

    private:
        //! Default constructor
        /** Shouldn't be called directly. Use 'GetUniqueInstance' instead.
        **/
        OCR_API explicit OcrEngine();

    private:
        class Impl;

        Impl *impl_;

    };
}