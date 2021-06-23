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

        inline int LoadModel(const ObjectEigenParams &params) {
            if (object_detector_ && object_detector_->getType() != params.objectDetectorType) {
                destroyObjectDetector();
            }

            if (!object_detector_) {
                switch (params.objectDetectorType) {
                    case YOLOV5:
                        object_detector_ = Yolov5Factory().createDetector();
                        break;
                    case MOBILENET_SSD:
                        object_detector_ = MobilenetSSDFactory().createDetector();
                        break;
                }

                if (!object_detector_ || object_detector_->load(params) != 0) {
                    std::cout << "load object detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (object_detector_->update(params) != 0) {
                std::cout << "update object detector model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            initialized_ = true;
            return 0;
        }

        inline int DetectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
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

    int ObjectEngine::loadModel(const ObjectEigenParams &params) {
        return impl_->LoadModel(params);
    }

    int ObjectEngine::detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        return impl_->DetectObject(img_src, objects);
    }

}

