#include "FaceEngine.h"
#include <iostream>

#include "../common/Singleton.h"
#include "detector/Detector.h"
#include "landmarker/LandMarker.h"
#include "recognizer/Recognizer.h"
#include "living/FaceAntiSpoofing.h"
#include "aligner/FaceAligner.h"
#include "tracker/Tracker.h"
#include "database/FaceDatabase.h"

namespace mirror {

    class FaceEngine::Impl {
    public:
        Impl() {
            tracker_ = new Tracker();
            aligner_ = new FaceAligner();
            database_ = new FaceDatabase();
            initialized_ = false;
        }

        ~Impl() {

            destroyFaceDetector();
            destroyFaceRecognizer();
            destroyFaceAntiSpoofing();
            destroyFaceLandMarker();

            if (tracker_) {
                delete tracker_;
                tracker_ = nullptr;
            }

            if (aligner_) {
                delete aligner_;
                aligner_ = nullptr;
            }

            if (database_) {
                delete database_;
                database_ = nullptr;
            }
        }

        void destroyFaceDetector() {
            if (detector_) {
                delete detector_;
                detector_ = nullptr;
            }
        }

        void destroyFaceRecognizer() {
            if (recognizer_) {
                delete recognizer_;
                recognizer_ = nullptr;
            }
        }

        void destroyFaceAntiSpoofing() {
            if (faceAntiSpoofing_) {
                delete faceAntiSpoofing_;
                faceAntiSpoofing_ = nullptr;
            }
        }

        void destroyFaceLandMarker() {
            if (landmarker_) {
                delete landmarker_;
                landmarker_ = nullptr;
            }
        }

        int initFaceDetector(const FaceEngineParams &params) {
            if (detector_ && detector_->getType() != params.faceDetectorType) {
                destroyFaceDetector();
            }

            if (!detector_) {
                switch (params.faceDetectorType) {
                    case RETINA_FACE:
                        detector_ = RetinafaceFactory().CreateDetector();
                        break;
                    case SCRFD_FACE:
                        detector_ = ScrfdFactory().CreateDetector();
                        break;
                    case CENTER_FACE:
                        detector_ = CenterfaceFactory().CreateDetector();
                        break;
                    case MTCNN_FACE:
                        detector_ = MtcnnFactory().CreateDetector();
                        break;
                    case ANTICOV_FACE:
                        detector_ = AnticovFactory().CreateDetector();
                        break;
                    default:
                        detector_ = RetinafaceFactory().CreateDetector();
                }

                if (!detector_ || detector_->load(params) != 0) {
                    std::cout << "load face detector failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (detector_->update(params) != 0) {
                std::cout << "update face detector model failed." << std::endl;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            return 0;
        }

        int initFaceRecognizer(const FaceEngineParams &params) {
            if (recognizer_ && recognizer_->getType() != params.faceRecognizerType) {
                destroyFaceRecognizer();
            }

            if (!recognizer_) {
                switch (params.faceRecognizerType) {
                    case ARC_FACE:
                        recognizer_ = MobilefacenetRecognizerFactory().CreateRecognizer();
                        break;
                    default:
                        recognizer_ = MobilefacenetRecognizerFactory().CreateRecognizer();
                }

                if (!recognizer_ || recognizer_->load(params) != 0) {
                    std::cout << "load face recognizer failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (recognizer_->update(params) != 0) {
                std::cout << "update face recognizer model failed." << std::endl;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            return 0;
        }

        int initFaceAntiSpoofing(const FaceEngineParams &params) {
            if (faceAntiSpoofing_ && faceAntiSpoofing_->getType() != params.faceAntiSpoofingType) {
                destroyFaceAntiSpoofing();
            }

            if (!faceAntiSpoofing_) {

                switch (params.faceAntiSpoofingType) {
                    case LIVE_FACE:
                        faceAntiSpoofing_ = LiveDetectorFactory().CreateFaceAntiSpoofing();
                        break;
                    default:
                        faceAntiSpoofing_ = LiveDetectorFactory().CreateFaceAntiSpoofing();
                }

                if (!faceAntiSpoofing_ || faceAntiSpoofing_->load(params) != 0) {
                    std::cout << "load face anti spoofing model failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (faceAntiSpoofing_->update(params) != 0) {
                std::cout << "update face anti spoofing model failed." << std::endl;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            return 0;

        }

        int initFaceLandMarker(const FaceEngineParams &params) {
            if (landmarker_ && landmarker_->getType() != params.faceLandMarkerType) {
                destroyFaceLandMarker();
            }

            if (!landmarker_) {
                switch (params.faceLandMarkerType) {
                    case INSIGHTFACE_LANDMARKER:
                        landmarker_ = InsightfaceLandMarkerFactory().CreateLandmarker();
                        break;
                    case ZQ_LANDMARKER:
                        landmarker_ = ZQLandMarkerFactory().CreateLandmarker();
                        break;
                    default:
                        landmarker_ = InsightfaceLandMarkerFactory().CreateLandmarker();
                }

                if (!landmarker_ || landmarker_->load(params) != 0) {
                    std::cout << "load face landmarker failed." << std::endl;
                    return ErrorCode::MODEL_LOAD_ERROR;
                }
            }

            if (landmarker_->update(params) != 0) {
                std::cout << "update face landmarker model failed." << std::endl;
                return ErrorCode::MODEL_UPDATE_ERROR;
            }

            return 0;

        }

        int LoadModel(const FaceEngineParams &params) {
            if (params.faceFeaturePath.empty()) {
                db_name_ = std::string(params.modelPath);
            } else {
                db_name_ = params.faceFeaturePath;
            }

            int errorCode = 0;
            if (params.faceDetectorEnabled) {
                if ((errorCode = initFaceDetector(params)) != 0) {
                    destroyFaceDetector();
                    initialized_ = false;
                    return errorCode;
                }
            } else {
                destroyFaceDetector();
            }

            if (params.faceRecognizerEnabled) {
                if ((errorCode = initFaceRecognizer(params)) != 0) {
                    destroyFaceRecognizer();
                    initialized_ = false;
                    return errorCode;
                }
            } else {
                destroyFaceRecognizer();
            }

            if (params.faceAntiSpoofingEnabled) {
                if ((errorCode = initFaceAntiSpoofing(params)) != 0) {
                    destroyFaceAntiSpoofing();
                    initialized_ = false;
                    return errorCode;
                }
            } else {
                destroyFaceAntiSpoofing();
            }

            if (params.faceLandMarkerEnabled) {
                if ((errorCode = initFaceLandMarker(params)) != 0) {
                    destroyFaceLandMarker();
                    initialized_ = false;
                    return errorCode;
                }
            } else {
                destroyFaceLandMarker();
            }

            PrintConfigurations(params);

            initialized_ = true;

            return 0;
        }

        inline void PrintConfigurations(const FaceEngineParams &params) const {
            std::cout << "--------------Face Configuration--------------" << std::endl;
            std::string configureInfo;
            configureInfo += std::string("GPU: ") + (params.gpuEnabled ? "True" : "False");
            configureInfo += std::string("\nVerbose: ") + (params.verbose ? "True" : "False");
            configureInfo += "\nModel path: " + params.modelPath;
            configureInfo += "\nFace feature database path: " + params.faceFeaturePath;

            configureInfo += std::string("\ndetector Enabled: ") +
                             (params.faceDetectorEnabled ? "True" : "False");
            configureInfo += std::string("\nrecognizer Enabled: ") +
                             (params.faceRecognizerEnabled ? "True" : "False");
            configureInfo += std::string("\nanti detector Enabled: ") +
                             (params.faceAntiSpoofingEnabled ? "True" : "False");
            configureInfo += std::string("\nlandmarker Enabled: ") +
                             (params.faceLandMarkerEnabled ? "True" : "False");
            configureInfo += std::string("\nthread number: ") + std::to_string(params.threadNum);

            if (detector_) {
                configureInfo += "\ndetector type: " + GetDetectorTypeName(detector_->getType());
            }

            if (recognizer_) {
                configureInfo += "\nrecognizer type: " + GetRecognizerTypeName(recognizer_->getType());
            }

            if (faceAntiSpoofing_) {
                configureInfo += "\nliving detector type: " +
                                 GetAntiSpoofingTypeName(faceAntiSpoofing_->getType());
            }

            if (landmarker_) {
                configureInfo += "\nlandmarker type: " +
                                 GetLandMarkerTypeName(landmarker_->getType());
            }

            std::cout << configureInfo << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        inline int Track(const std::vector<FaceInfo> &curr_faces, std::vector<TrackedFaceInfo> &faces) {
            if (!initialized_ || !tracker_) {
                std::cout << "face tracker model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return tracker_->track(curr_faces, faces);
        }

        inline bool DetectLivingFace(const cv::Mat &img_src, const cv::Rect &box, float &livingScore) {
            if (!initialized_ || !faceAntiSpoofing_) {
                std::cout << "face anti spoofing model uninitialized!" << std::endl;
                return true;
            }
            return faceAntiSpoofing_->detect(img_src, box, livingScore);
        }

        inline int DetectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) {
            if (!initialized_ || !detector_) {
                std::cout << "face detector model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return detector_->detect(img_src, faces);
        }

        inline int ExtractKeypoints(const cv::Mat &img_src,
                                    const cv::Rect &face, std::vector<cv::Point2f> &keypoints) {
            if (!initialized_ || !landmarker_) {
                std::cout << "face landmark model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return landmarker_->extract(img_src, face, keypoints);
        }

        inline int AlignFace(const cv::Mat &img_src, const std::vector<cv::Point2f> &keypoints,
                             cv::Mat &face_aligned) {
            if (!initialized_ || !aligner_) {
                std::cout << "face aligner model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return aligner_->alignFace(img_src, keypoints, face_aligned);
        }

        inline int ExtractFeature(const cv::Mat &img_face, std::vector<float> &feat) {
            if (!initialized_ || !recognizer_) {
                std::cout << "face recognizer model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return recognizer_->extract(img_face, feat);
        }

        inline int Insert(const std::vector<float> &feat, const std::string &name) {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return static_cast<int>(database_->Insert(feat, name));
        }

        inline int Find(std::vector<std::string> &names) const {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return database_->Find(names);
        }

        inline int Delete(const std::string &name) {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return database_->Delete(name);
        }

        inline int Clear() {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            database_->Clear();
            return 0;
        }

        inline int QueryTop(const std::vector<float> &feat, QueryResult &query_result) {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return database_->QueryTop(feat, query_result);
        }

        inline int Save() const {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return database_->Save(db_name_.c_str());
        }

        inline int Load() {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return database_->Load(db_name_.c_str());
        }

    private:
        bool initialized_;
        std::string db_name_;
        FaceAntiSpoofing *faceAntiSpoofing_ = nullptr;
        Detector *detector_ = nullptr;
        LandMarker *landmarker_ = nullptr;
        Recognizer *recognizer_ = nullptr;
        FaceAligner *aligner_ = nullptr;
        Tracker *tracker_ = nullptr;
        FaceDatabase *database_ = nullptr;
    };

    //! Unique instance of ecvOptions
    static Singleton<FaceEngine> s_engine;

    FaceEngine *FaceEngine::GetInstancePtr() {
        if (!s_engine.instance) {
            s_engine.instance = new FaceEngine();
        }
        return s_engine.instance;
    }

    FaceEngine &FaceEngine::GetInstance() {
        if (!s_engine.instance) {
            s_engine.instance = new FaceEngine();
        }

        return *s_engine.instance;
    }

    void FaceEngine::ReleaseInstance() {
        s_engine.release();
    }

    FaceEngine::FaceEngine() {
        impl_ = new FaceEngine::Impl();
    }

    void FaceEngine::destroyEngine() {
        ReleaseInstance();
    }

    FaceEngine::~FaceEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int FaceEngine::loadModel(const FaceEngineParams &params) {
        return impl_->LoadModel(params);
    }

    int FaceEngine::track(const std::vector<FaceInfo> &curr_faces,
                          std::vector<TrackedFaceInfo> &faces) {
        return impl_->Track(curr_faces, faces);
    }

    bool FaceEngine::detectLivingFace(const cv::Mat &img_src, const cv::Rect &box, float &livingScore) const {
        return impl_->DetectLivingFace(img_src, box, livingScore);
    }

    int FaceEngine::detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const {
        return impl_->DetectFace(img_src, faces);
    }

    int FaceEngine::extractKeypoints(const cv::Mat &img_src,
                                     const cv::Rect &face,
                                     std::vector<cv::Point2f> &keypoints) const {
        return impl_->ExtractKeypoints(img_src, face, keypoints);
    }

    int FaceEngine::alignFace(const cv::Mat &img_src,
                              const std::vector<cv::Point2f> &keypoints,
                              cv::Mat &face_aligned) const {
        return impl_->AlignFace(img_src, keypoints, face_aligned);
    }

    int FaceEngine::extractFeature(const cv::Mat &img_face, std::vector<float> &feat) const {
        return impl_->ExtractFeature(img_face, feat);
    }

    int FaceEngine::Save() const {
        return impl_->Save();
    }

    int FaceEngine::Load() {
        return impl_->Load();
    }

    int FaceEngine::Clear() {
        return impl_->Clear();
    }

    int FaceEngine::Delete(const std::string &name) {
        return impl_->Delete(name);
    }

    int FaceEngine::Insert(const std::vector<float> &feat, const std::string &name) {
        return impl_->Insert(feat, name);
    }

    int FaceEngine::QueryTop(const std::vector<float> &feat, QueryResult &query_result) const {
        return impl_->QueryTop(feat, query_result);
    }

    int FaceEngine::Find(std::vector<std::string> &names) const {
        return impl_->Find(names);
    }
}