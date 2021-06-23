#include "ClassifierEngine.h"
#include "classifier/Classifier.h"
#include "../common/Singleton.h"

#include <string>
#include <iostream>
#include <opencv2/core.hpp>

namespace mirror {

    class ClassifierEngine::Impl {
    public:

        Impl() {
            initialized_ = false;
        }

        ~Impl() {
            destroyClassifier();
        }

        void destroyClassifier() {
            if (classifier_) {
                delete classifier_;
                classifier_ = nullptr;
            }
        }

        int LoadModel(const ClassifierEigenParams &params) {
            if (classifier_ && classifier_->getType() != params.classifierType) {
                destroyClassifier();
            }

            if (!classifier_) {
                switch (params.classifierType) {
                    case MOBILE_NET:
                        classifier_ = MobilenetFactory().createClassifier();
                        break;
                }

                if (!classifier_ || classifier_->load(params) != 0) {
                    std::cout << "load object classifier failed." << std::endl;
                    initialized_ = false;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (classifier_->update(params) != 0) {
                std::cout << "update object classifier model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            initialized_ = true;
            return 0;
        }

        int Classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
            if (!initialized_ || !classifier_) {
                std::cout << "object classifier model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return classifier_->classify(img_src, images);
        }

    private:
        Classifier *classifier_ = nullptr;
        bool initialized_;
    };

    //! Unique instance of ecvOptions
    static Singleton<ClassifierEngine> s_engine;

    ClassifierEngine *ClassifierEngine::GetInstancePtr() {
        if (!s_engine.instance) {
            s_engine.instance = new ClassifierEngine();
        }
        return s_engine.instance;
    }

    ClassifierEngine &ClassifierEngine::GetInstance() {
        if (!s_engine.instance) {
            s_engine.instance = new ClassifierEngine();
        }

        return *s_engine.instance;
    }

    void ClassifierEngine::ReleaseInstance() {
        s_engine.release();
    }

    ClassifierEngine::ClassifierEngine() {
        impl_ = new ClassifierEngine::Impl();
    }

    void ClassifierEngine::destroyEngine() {
        ReleaseInstance();
    }

    ClassifierEngine::~ClassifierEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int ClassifierEngine::loadModel(const ClassifierEigenParams &params) {
        return impl_->LoadModel(params);
    }

    int ClassifierEngine::classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
        return impl_->Classify(img_src, images);
    }

}


