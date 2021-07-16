#include "ClassifierEngine.h"
#include "classifiers/Classifier.h"
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

        inline void PrintConfigurations(const ClassifierEngineParams &params) const {
            std::cout << "--------------Classifier Configuration--------------" << std::endl;
            std::string configureInfo;
            configureInfo += std::string("GPU: ") + (params.gpuEnabled ? "True" : "False");
            configureInfo += std::string("\nVerbose: ") + (params.verbose ? "True" : "False");
            configureInfo += "\nModel path: " + params.modelPath;

            configureInfo += std::string("\ntopK: ") + std::to_string(params.topK);
            configureInfo += std::string("\nthread number: ") + std::to_string(params.threadNum);

            if (classifier_) {
                configureInfo += "\nclassifiers type: " + GetClassifierTypeName(classifier_->getType());
            }

            std::cout << configureInfo << std::endl;
            std::cout << "---------------------------------------------------" << std::endl;
        }

        int LoadModel(const ClassifierEngineParams &params) {
            if (classifier_ && classifier_->getType() != params.classifierType) {
                destroyClassifier();
            }

            if (!classifier_) {
                switch (params.classifierType) {
                    case MOBILE_NET:
                        classifier_ = MobilenetFactory().createClassifier();
                        break;
                    case SQUEEZE_NET:
                        classifier_ = SqueezeNetFactory().createClassifier();
                        break;
                    default:
                        std::cout << "unsupported model type!." << std::endl;
                        break;
                }

                if (!classifier_ || classifier_->load(params) != ErrorCode::SUCCESS) {
                    std::cout << "load object classifiers failed." << std::endl;
                    initialized_ = false;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (classifier_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update object classifiers model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        inline int UpdateModel(const ClassifierEngineParams &params) {
            if (!classifier_ || classifier_->getType() != params.classifierType) {
                return LoadModel(params);
            }

            if (classifier_->update(params) != ErrorCode::SUCCESS) {
                std::cout << "update object classifiers model failed." << std::endl;
                initialized_ = false;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            PrintConfigurations(params);

            initialized_ = true;
            return ErrorCode::SUCCESS;
        }

        int Classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
            if (!initialized_ || !classifier_) {
                std::cout << "object classifiers model uninitialized!" << std::endl;
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

    int ClassifierEngine::loadModel(const ClassifierEngineParams &params) {
        return impl_->LoadModel(params);
    }

    int ClassifierEngine::updateModel(const ClassifierEngineParams &params) {
        return impl_->UpdateModel(params);
    }

    int ClassifierEngine::classify(const cv::Mat &img_src, std::vector<ImageInfo> &images) const {
        return impl_->Classify(img_src, images);
    }

}


