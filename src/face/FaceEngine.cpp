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
                        std::cout << "unsupported face detector model type!." << std::endl;
                        break;
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
                        std::cout << "unsupported face recognizer model type!." << std::endl;
                        break;
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
                        std::cout << "unsupported face living detector model type!." << std::endl;
                        break;
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
                        std::cout << "unsupported face landmarker model type!." << std::endl;
                        break;
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

        inline int Track(const std::vector<FaceInfo> &currFaces, std::vector<TrackedFaceInfo> &faces) {
            if (!initialized_ || !tracker_) {
                std::cout << "face tracker model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return tracker_->track(currFaces, faces);
        }

        inline bool DetectLivingFace(const cv::Mat &imgSrc, float &livingScore) const {
            if (!initialized_ || !detector_ || !faceAntiSpoofing_) {
                std::cout << "face anti spoofing model uninitialized!" << std::endl;
                return false;
            }
            std::vector<FaceInfo> faces;
            DetectFace(imgSrc, faces);
            if (faces.empty()) {
                std::cout << "Cannot detect any face!" << std::endl;
                return false;
            }
            return faceAntiSpoofing_->detect(imgSrc, faces[0].location_, livingScore);
        }

        inline bool DetectLivingFace(const cv::Mat &imgSrc, const cv::Rect &box, float &livingScore) const {
            if (!initialized_ || !faceAntiSpoofing_) {
                std::cout << "face anti spoofing model uninitialized!" << std::endl;
                return false;
            }
            return faceAntiSpoofing_->detect(imgSrc, box, livingScore);
        }

        inline int VerifyFace(const cv::Mat &imgSrc, VerificationResult &result,
                              bool livingEnabled = false) const {
            if (!initialized_ || !aligner_ || !detector_ || !recognizer_ || !database_) {
                std::cout << "face detector, recognizer model or database uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }

            if (livingEnabled && !faceAntiSpoofing_) {
                std::cout << "face anti spoofing model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }

            std::vector<FaceInfo> faces;
            DetectFace(imgSrc, faces);
            if (faces.empty()) {
                std::cout << "Cannot detect any face!" << std::endl;
                return ErrorCode::NOT_FOUND_ERROR;
            }

            bool is_living = true;
            float livingScore = 1.0f;
            if (livingEnabled) {
                is_living = DetectLivingFace(imgSrc, faces.at(0).location_, livingScore);
            }

            if (is_living) {
                // align face
                cv::Mat faceAligned;
                std::vector<cv::Point2f> keyPoints;
                // only register first face
                ConvertKeyPoints(faces.at(0).keypoints_, 5, keyPoints);
                AlignFace(imgSrc, keyPoints, faceAligned);

                // extract feature
                std::vector<float> feat;
                int flag = ExtractFeature(faceAligned, feat);
                if (flag != 0 || feat.size() != kFaceFeatureDim) {
                    return ErrorCode::DIMENSION_MISS_MATCH_ERROR;
                }

                // compare feature
                QueryResult query_result;
                query_result.sim_ = 0.0f;
                query_result.name_ = "stranger";
                flag = QueryTop(feat, query_result);
                if (flag == ErrorCode::EMPTY_DATA_ERROR) {
                    std::cout << "face database is empty, please register first!" << std::endl;
                    return ErrorCode::EMPTY_DATA_ERROR;
                }

                result.sim = query_result.sim_;
                result.name = query_result.name_;

                return ErrorCode::SUCCESS;
            }

            return ErrorCode::SUCCESS;
        }

        inline int VerifyFace(const cv::Mat &imgSrc,
                              const std::vector<cv::Point2f> &keyPoints,
                              VerificationResult &result) const {
            if (!initialized_ || !aligner_ || !recognizer_ || !database_) {
                std::cout << "face detector, recognizer model or database uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }

            std::vector<FaceInfo> faces;
            DetectFace(imgSrc, faces);
            if (faces.empty()) {
                std::cout << "Cannot detect any face!" << std::endl;
                return ErrorCode::NOT_FOUND_ERROR;
            }

            // align face
            cv::Mat faceAligned;
            AlignFace(imgSrc, keyPoints, faceAligned);

            // extract feature
            std::vector<float> feat;
            int flag = ExtractFeature(faceAligned, feat);
            if (flag != 0 || feat.size() != kFaceFeatureDim) {
                return ErrorCode::DIMENSION_MISS_MATCH_ERROR;
            }

            // compare feature
            QueryResult query_result;
            query_result.sim_ = 0.0f;
            query_result.name_ = "stranger";
            flag = QueryTop(feat, query_result);
            if (flag == ErrorCode::EMPTY_DATA_ERROR) {
                std::cout << "face database is empty, please register first!" << std::endl;
                return ErrorCode::EMPTY_DATA_ERROR;
            }

            result.sim = query_result.sim_;
            result.name = query_result.name_;

            return ErrorCode::SUCCESS;
        }

        inline int RegisterFace(const cv::Mat &imgSrc, const std::string &name) {
            if (!initialized_ || !detector_ || !recognizer_ || !database_) {
                std::cout << "face detector, recognizer model or database uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }

            if (database_->IsEmpty()) {
                std::cout << "database unloaded!" << std::endl;
                if (Load() != 0) {
                    std::cout << "database load failed!" << std::endl;
                    return ErrorCode::UNINITIALIZED_ERROR;
                }
            }

            std::vector<FaceInfo> faces;
            DetectFace(imgSrc, faces);
            if (faces.empty()) {
                std::cout << "Cannot detect any face!" << std::endl;
                return ErrorCode::NOT_FOUND_ERROR;
            }

            // align face
            cv::Mat faceAligned;
            std::vector<cv::Point2f> keyPoints;
            // only register first face
            ConvertKeyPoints(faces.at(0).keypoints_, 5, keyPoints);
            AlignFace(imgSrc, keyPoints, faceAligned);

            // extract feature
            std::vector<float> feat;
            int flag = ExtractFeature(faceAligned, feat);
            if (flag != 0 || feat.size() != kFaceFeatureDim) {
                return flag;
            }

            Insert(feat, name);
            if (Save() != 0) {
                std::cout << "Save face database failed!" << std::endl;
                return ErrorCode::DATABASE_UPDATE_ERROR;
            }

            return ErrorCode::SUCCESS;
        }

        inline int DetectFace(const cv::Mat &imgSrc, std::vector<FaceInfo> &faces) const {
            if (!initialized_ || !detector_) {
                std::cout << "face detector model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return detector_->detect(imgSrc, faces);
        }

        inline int ExtractKeypoints(const cv::Mat &imgSrc,
                                    const cv::Rect &box, std::vector<cv::Point2f> &keypoints) {
            if (!initialized_ || !landmarker_) {
                std::cout << "face landmark model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return landmarker_->extract(imgSrc, box, keypoints);
        }

        inline int AlignFace(const cv::Mat &imgSrc,
                             const std::vector<cv::Point2f> &keypoints,
                             cv::Mat &faceAligned) const {
            if (!initialized_ || !aligner_) {
                std::cout << "face aligner model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return aligner_->alignFace(imgSrc, keypoints, faceAligned);
        }

        inline int ExtractFeature(const cv::Mat &imgSrc, std::vector<float> &feat) const {
            if (!initialized_ || !recognizer_) {
                std::cout << "face recognizer model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return recognizer_->extract(imgSrc, feat);
        }

        inline int Insert(const std::vector<float> &feat, const std::string &name) {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            if (database_->IsEmpty()) {
                std::cout << "database unloaded!" << std::endl;
                if (Load() != 0) {
                    std::cout << "database load failed!" << std::endl;
                    return ErrorCode::UNINITIALIZED_ERROR;
                }
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
            if (database_->IsEmpty()) {
                std::cout << "database unloaded!" << std::endl;
                if (Load() != 0) {
                    std::cout << "database load failed!" << std::endl;
                    return ErrorCode::UNINITIALIZED_ERROR;
                }
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

        inline int QueryTop(const std::vector<float> &feat, QueryResult &queryResult) const {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
            return database_->QueryTop(feat, queryResult);
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

        inline bool databaseEmpty() const {
            if (!initialized_ || !database_) {
                std::cout << "face database model uninitialized!" << std::endl;
                return false;
            }
            return database_->IsEmpty();
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

    int FaceEngine::track(const std::vector<FaceInfo> &currFaces,
                          std::vector<TrackedFaceInfo> &faces) {
        return impl_->Track(currFaces, faces);
    }

    int FaceEngine::detectFace(const cv::Mat &imgSrc, std::vector<FaceInfo> &faces) const {
        return impl_->DetectFace(imgSrc, faces);
    }

    int FaceEngine::extractKeypoints(const cv::Mat &imgSrc,
                                     const cv::Rect &box,
                                     std::vector<cv::Point2f> &keypoints) const {
        return impl_->ExtractKeypoints(imgSrc, box, keypoints);
    }

    int FaceEngine::alignFace(const cv::Mat &imgSrc,
                              const std::vector<cv::Point2f> &keypoints,
                              cv::Mat &faceAligned) const {
        return impl_->AlignFace(imgSrc, keypoints, faceAligned);
    }

    int FaceEngine::extractFeature(const cv::Mat &imgSrc, std::vector<float> &feat) const {
        return impl_->ExtractFeature(imgSrc, feat);
    }

    int FaceEngine::registerFace(const cv::Mat &imgSrc, const std::string &name) {
        return impl_->RegisterFace(imgSrc, name);
    }

    bool FaceEngine::detectLivingFace(const cv::Mat &imgSrc, const cv::Rect &box, float &livingScore) const {
        return impl_->DetectLivingFace(imgSrc, box, livingScore);
    }

    bool FaceEngine::detectLivingFace(const cv::Mat &imgSrc, float &livingScore) const {
        return impl_->DetectLivingFace(imgSrc, livingScore);
    }

    int FaceEngine::verifyFace(const cv::Mat &imgSrc, VerificationResult &result, bool livingEnabled) const {
        return impl_->VerifyFace(imgSrc, result, livingEnabled);
    }

    int FaceEngine::verifyFace(const cv::Mat &imgSrc, const std::vector<cv::Point2f> &keyPoints,
                               VerificationResult &result) const {
        return impl_->VerifyFace(imgSrc, keyPoints, result);
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

    int FaceEngine::QueryTop(const std::vector<float> &feat, QueryResult &queryResult) const {
        if (impl_->databaseEmpty()) {
            std::cout << "database unloaded!" << std::endl;
            if (impl_->Load() != 0) {
                std::cout << "database load failed!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
        }
        return impl_->QueryTop(feat, queryResult);
    }

    int FaceEngine::Find(std::vector<std::string> &names) const {
        if (impl_->databaseEmpty()) {
            std::cout << "database unloaded!" << std::endl;
            if (impl_->Load() != 0) {
                std::cout << "database load failed!" << std::endl;
                return ErrorCode::UNINITIALIZED_ERROR;
            }
        }
        return impl_->Find(names);
    }

}