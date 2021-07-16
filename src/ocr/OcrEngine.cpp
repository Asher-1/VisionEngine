#include "OcrEngine.h"
#include "detectors/TextDetector.h"
#include "recognizers/TextRecognizer.h"
#include "../common/Singleton.h"

#include <string>
#include <iostream>
#include <opencv2/core.hpp>

namespace mirror {

    class OcrEngine::Impl {
    public:

        Impl() {
            initialized_ = false;
        }

        ~Impl() {
            destroyTextDetector();
            destroyTextRecognizer();
        }

        void destroyTextDetector() {
            if (detector_) {
                delete detector_;
                detector_ = nullptr;
            }
        }

        void destroyTextRecognizer() {
            if (recognizer_) {
                delete recognizer_;
                recognizer_ = nullptr;
            }
        }

        inline void PrintConfigurations(const OcrEngineParams &params) const {
            std::cout << "--------------OCR Configuration--------------" << std::endl;
            std::string configureInfo;
            configureInfo += std::string("GPU: ") + (params.gpuEnabled ? "True" : "False");
            configureInfo += std::string("\nVerbose: ") + (params.verbose ? "True" : "False");
            configureInfo += "\nModel path: " + params.modelPath;
            configureInfo += std::string("\nthread number: ") + std::to_string(params.threadNum);

            if (detector_) {
                configureInfo += "\nText detector type: " + GetTextDetectorTypeName(detector_->getType());
            }

            if (recognizer_) {
                configureInfo += "\nText recognizer type: " + GetTextRecognizerTypeName(recognizer_->getType());
            }

            std::cout << configureInfo << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
        }

        int initTextDetector(const OcrEngineParams &params) {
            if (detector_ && detector_->getType() != params.textDetectorType) {
                destroyTextDetector();
            }

            if (!detector_) {
                switch (params.textDetectorType) {
                    case DB_NET:
                        detector_ = DBNetFactory().create();
                        break;
                    default:
                        std::cout << "unsupported model type!." << std::endl;
                        break;
                }

                if (!detector_ || detector_->load(params) != ErrorCode::SUCCESS) {
                    std::cout << "load text detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (detector_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update text detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            return ErrorCode::SUCCESS;
        }

        int initTextRecognizer(const OcrEngineParams &params) {
            if (recognizer_ && recognizer_->getType() != params.textRecognizerType) {
                destroyTextRecognizer();
            }

            if (!recognizer_) {
                switch (params.textRecognizerType) {
                    case CRNN_NET:
                        recognizer_ = CRNNNetFactory().create();
                        break;
                    default:
                        std::cout << "unsupported model type!." << std::endl;
                        break;
                }

                if (!recognizer_ || recognizer_->load(params) != ErrorCode::SUCCESS) {
                    std::cout << "load text recognizer failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (recognizer_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update text recognizer model failed." << std::endl;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            return ErrorCode::SUCCESS;
        }

        int LoadModel(const OcrEngineParams &params) {
            if (detector_ && detector_->getType() != params.textDetectorType) {
                destroyTextDetector();
            }

            int errorCode = ErrorCode::SUCCESS;
            if ((errorCode = initTextDetector(params)) != ErrorCode::SUCCESS) {
                destroyTextDetector();
                initialized_ = false;
                return errorCode;
            }

            if ((errorCode = initTextRecognizer(params)) != ErrorCode::SUCCESS) {
                destroyTextRecognizer();
                initialized_ = false;
                return errorCode;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        inline int UpdateModel(const OcrEngineParams &params) {
            if (!detector_ || detector_->getType() != params.textDetectorType ||
                !recognizer_ || recognizer_->getType() != params.textRecognizerType) {
                return LoadModel(params);
            }

            if (detector_->update(params) != ErrorCode::SUCCESS ||
                recognizer_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update ocr model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);
            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        int detect(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const {
            if (!initialized_ || !detector_) {
                std::cout << "ocr model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return detector_->detect(img_src, textBoxes);
        }

        int Recognize(const cv::Mat &img_src,
                      const std::vector<TextBox> &textBoxes,
                      std::vector<OCRResult> &ocrResults) const {
            if (!initialized_ || !recognizer_) {
                std::cout << "ocr model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return recognizer_->recognize(img_src, textBoxes, ocrResults);
        }

    private:
        TextDetector *detector_ = nullptr;
        TextRecognizer *recognizer_ = nullptr;
        bool initialized_;
    };

    //! Unique instance of ecvOptions
    static Singleton<OcrEngine> s_engine;

    OcrEngine *OcrEngine::GetInstancePtr() {
        if (!s_engine.instance) {
            s_engine.instance = new OcrEngine();
        }
        return s_engine.instance;
    }

    OcrEngine &OcrEngine::GetInstance() {
        if (!s_engine.instance) {
            s_engine.instance = new OcrEngine();
        }

        return *s_engine.instance;
    }

    void OcrEngine::ReleaseInstance() {
        s_engine.release();
    }

    OcrEngine::OcrEngine() {
        impl_ = new OcrEngine::Impl();
    }

    void OcrEngine::destroyEngine() {
        ReleaseInstance();
    }

    OcrEngine::~OcrEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int OcrEngine::loadModel(const OcrEngineParams &params) {
        return impl_->LoadModel(params);
    }

    int OcrEngine::updateModel(const OcrEngineParams &params) {
        return impl_->UpdateModel(params);
    }

    int OcrEngine::detectText(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const {
        return impl_->detect(img_src, textBoxes);
    }

    int OcrEngine::recognizeText(const cv::Mat &img_src,
                                 const std::vector<TextBox> &textBoxes,
                                 std::vector<OCRResult> &ocrResults) const {
        return impl_->Recognize(img_src, textBoxes, ocrResults);
    }

}


