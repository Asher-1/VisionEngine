#include "ObjectEngine.h"
#include "detectors/ObjectDetector.h"
#include "../common/Singleton.h"

#include <string>
#include <iostream>

namespace mirror {
    class ObjectEngine::Impl {
    public:
        Impl() {
            initialized_ = false;
        }

        ~Impl() {
            destroyObjectDetector();
        }

        void destroyObjectDetector() {
            if (object_detector_) {
                delete object_detector_;
                object_detector_ = nullptr;
            }
        }

        inline void PrintConfigurations(const ObjectEngineParams &params) const {
            std::cout << "--------------Object Configuration--------------" << std::endl;
            std::string configureInfo;
            configureInfo += std::string("GPU: ") + (params.gpuEnabled ? "True" : "False");
            configureInfo += std::string("\nVerbose: ") + (params.verbose ? "True" : "False");
            configureInfo += "\nModel path: " + params.modelPath;

            configureInfo += std::string("\nnmsThreshold: ") + std::to_string(params.nmsThreshold);
            if (params.objectDetectorType == ObjectDetectorType::YOLOV4) {
                std::string modelName;
                if (params.modeType == ErrorCode::SUCCESS) {
                    modelName = "yolov4-tiny-opt";
                } else if (params.modeType == 1) {
                    modelName = "MobileNetV2-YOLOv3-Nano-coco";
                } else if (params.modeType == 2) {
                    modelName = "yolo-fastest-opt";
                }
                configureInfo += std::string("\nmodeType: ") + modelName;
            }
            configureInfo += std::string("\nscoreThreshold: ") + std::to_string(params.scoreThreshold);
            configureInfo += std::string("\nthread number: ") + std::to_string(params.threadNum);

            if (object_detector_) {
                configureInfo += "\nObject detector type: " + GetObjectDetectorTypeName(object_detector_->getType());
            }

            std::cout << configureInfo << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
        }

        inline int UpdateModel(const ObjectEngineParams &params) {
            if (!object_detector_ || object_detector_->getType() != params.objectDetectorType) {
                return LoadModel(params);
            }

            if (object_detector_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update object detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        inline int LoadModel(const ObjectEngineParams &params) {
            if (object_detector_ && object_detector_->getType() != params.objectDetectorType) {
                destroyObjectDetector();
            }

            if (!object_detector_) {
                switch (params.objectDetectorType) {
                    case YOLOV4:
                        object_detector_ = Yolov4Factory().createDetector();
                        break;
                    case YOLOV5:
                        object_detector_ = Yolov5Factory().createDetector();
                        break;
                    case NANO_DET:
                        object_detector_ = NanoDetFactory().createDetector();
                        break;
                    case MOBILENET_SSD:
                        object_detector_ = MobilenetSSDFactory().createDetector();
                        break;
                    default:
                        std::cout << "unsupported model type!." << std::endl;
                        break;
                }

                if (!object_detector_ || object_detector_->load(params) != ErrorCode::SUCCESS) {
                    std::cout << "load object detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (object_detector_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update object detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        inline int Detect(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
            if (!initialized_ || !object_detector_) {
                std::cout << "object detector model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return object_detector_->detect(img_src, objects);
        }

    private:
        ObjectDetector *object_detector_ = nullptr;
        bool initialized_;
    };


    //! Unique instance of ecvOptions
    static Singleton<ObjectEngine> s_engine;

    ObjectEngine *ObjectEngine::GetInstancePtr() {
        if (!s_engine.instance) {
            s_engine.instance = new ObjectEngine();
        }
        return s_engine.instance;
    }

    ObjectEngine &ObjectEngine::GetInstance() {
        if (!s_engine.instance) {
            s_engine.instance = new ObjectEngine();
        }

        return *s_engine.instance;
    }

    void ObjectEngine::ReleaseInstance() {
        s_engine.release();
    }

    ObjectEngine::ObjectEngine() {
        impl_ = new ObjectEngine::Impl();
    }

    void ObjectEngine::destroyEngine() {
        ReleaseInstance();
    }

    ObjectEngine::~ObjectEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int ObjectEngine::loadModel(const ObjectEngineParams &params) {
        return impl_->LoadModel(params);
    }

    int ObjectEngine::updateModel(const ObjectEngineParams &params) {
        return impl_->UpdateModel(params);
    }

    int ObjectEngine::detect(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        return impl_->Detect(img_src, objects);
    }

}

