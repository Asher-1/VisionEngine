#include "SegmentEngine.h"
#include "detectors/SegmentDetector.h"
#include "../common/Singleton.h"

#include <string>
#include <iostream>

namespace mirror {
    class SegmentEngine::Impl {
    public:
        Impl() {
            initialized_ = false;
        }

        ~Impl() {
            destroySegmentDetector();
        }

        void destroySegmentDetector() {
            if (segment_detector_) {
                delete segment_detector_;
                segment_detector_ = nullptr;
            }
        }

        inline void PrintConfigurations(const SegmentEngineParams &params) const {
            std::cout << "--------------Segment Configuration--------------" << std::endl;
            std::string configureInfo;
            configureInfo += std::string("GPU: ") + (params.gpuEnabled ? "True" : "False");
            configureInfo += std::string("\nVerbose: ") + (params.verbose ? "True" : "False");
            configureInfo += "\nModel path: " + params.modelPath;

            configureInfo += std::string("\nnmsThreshold: ") + std::to_string(params.nmsThreshold);
            configureInfo += std::string("\nscoreThreshold: ") + std::to_string(params.scoreThreshold);
            configureInfo += std::string("\nthread number: ") + std::to_string(params.threadNum);

            if (segment_detector_) {
                configureInfo += "\nSegment detector type: " + GetSegmentTypeName(segment_detector_->getType());
            }

            std::cout << configureInfo << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;
        }

        inline int LoadModel(const SegmentEngineParams &params) {
            if (segment_detector_ && segment_detector_->getType() != params.segmentType) {
                destroySegmentDetector();
            }

            if (!segment_detector_) {
                switch (params.segmentType) {
                    case YOLACT_SEG:
                        segment_detector_ = YolactSegFactory().createDetector();
                        break;
                    case MOBILENETV3_SEG:
                        segment_detector_ = MobileNetV3SegFactory().createDetector();
                        break;
                }

                if (!segment_detector_ || segment_detector_->load(params) != 0) {
                    std::cout << "load segment detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (segment_detector_->update(params) != 0) {
                std::cout << "update segment detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::MODEL_UPDATE_ERROR;
        }

        inline int UpdateModel(const SegmentEngineParams &params) {
            if (!segment_detector_ || segment_detector_->getType() != params.segmentType) {
                return LoadModel(params);
            }

            if (segment_detector_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update segment detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        inline int Detect(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const {
            if (!initialized_ || !segment_detector_) {
                std::cout << "segment detector model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return segment_detector_->detect(img_src, segments);
        }

    private:
        SegmentDetector *segment_detector_ = nullptr;
        bool initialized_;
    };

    //! Unique instance of ecvOptions
    static Singleton<SegmentEngine> s_engine;

    SegmentEngine *SegmentEngine::GetInstancePtr() {
        if (!s_engine.instance) {
            s_engine.instance = new SegmentEngine();
        }
        return s_engine.instance;
    }

    SegmentEngine &SegmentEngine::GetInstance() {
        if (!s_engine.instance) {
            s_engine.instance = new SegmentEngine();
        }

        return *s_engine.instance;
    }

    void SegmentEngine::ReleaseInstance() {
        s_engine.release();
    }

    SegmentEngine::SegmentEngine() {
        impl_ = new SegmentEngine::Impl();
    }

    void SegmentEngine::destroyEngine() {
        ReleaseInstance();
    }

    SegmentEngine::~SegmentEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int SegmentEngine::loadModel(const SegmentEngineParams &params) {
        return impl_->LoadModel(params);
    }

    int SegmentEngine::updateModel(const SegmentEngineParams &params) {
        return impl_->UpdateModel(params);
    }

    int SegmentEngine::detect(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const {
        return impl_->Detect(img_src, segments);
    }

}

