#define FACE_EXPORTS

#include "FaceEngine.h"
#include "VisionTools.h"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace mirror;

static const bool use_living = true;
static bool use_gpu = false;
static std::string img_path = "../../data/images/4.jpg";
static std::string model_path = "../../data/models";
static FaceDetectorType detectorModelType = FaceDetectorType::RETINA_FACE;
static FaceRecognizerType recognizerModelType = FaceRecognizerType::ARC_FACE;

int TestDetector(int argc, char **argv) {
    std::string result_path = "../../data/images/detector_result.jpg";
    if (argc >= 2) {
        result_path = "detector_result.jpg";
    }

    std::cout << "Face Detection Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = false;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    utility::DrawFaces(img_src, faces, true);
    cv::imwrite(result_path, img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();

    return 0;
}

int TestLandmark(int argc, char *argv[]) {
    std::string result_path = "../../data/images/landmark_result.jpg";
    if (argc >= 2) {
        result_path = "landmark_result.jpg";
    }

    std::cout << "Face LandMark Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceLandMarkerEnabled = true;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = false;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        cv::Rect face = faces.at(i).location_;
        std::vector<cv::Point2f> keypoints;
        face_engine->extractKeypoints(img_src, face, keypoints);
        utility::DrawKeyPoints(img_src, keypoints, 1, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    cv::imwrite(result_path, img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();

    return 0;

}

int TestAlignFace(int argc, char *argv[]) {
    std::cout << "Face Alignment Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);

    const bool use_landmark = false;

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceLandMarkerEnabled = use_landmark;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = false;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        cv::Rect face = faces.at(i).location_;
        std::vector<cv::Point2f> keypoints;
        if (params.faceLandMarkerEnabled) {
            face_engine->extractKeypoints(img_src, face, keypoints);
        } else {
            ConvertKeyPoints(faces.at(i).keypoints_, 5, keypoints);
        }
        cv::Mat face_aligned;
        face_engine->alignFace(img_src, keypoints, face_aligned);
        std::string name = std::to_string(i) + ".jpg";
        cv::imwrite(name.c_str(), face_aligned);
        utility::DrawKeyPoints(img_src, keypoints, 1, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();

    return 0;
}

int TestMask(int argc, char *argv[]) {
    std::string result_path = "../../data/images/mask_result.jpg";
    if (argc >= 2) {
        result_path = "mask_result.jpg";
    }

    std::cout << "Face Mask Test......" << std::endl;
    const char *mask_file = "../../data/images/mask3.jpg";
    cv::Mat img_src = cv::imread(mask_file);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = false;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    params.faceDetectorType = FaceDetectorType::ANTICOV_FACE;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    int num_face = static_cast<int>(faces.size());
    for (int i = 0; i < num_face; ++i) {
        char text[256];
        if (faces[i].mask_) {
            sprintf(text, "%s %.1f%%", "mask", faces[i].score_ * 100);
            cv::rectangle(img_src, faces[i].location_, cv::Scalar(0, 255, 0), 2);
        } else {
            sprintf(text, "%s %.1f%%", "no mask", faces[i].score_ * 100);
            cv::rectangle(img_src, faces[i].location_, cv::Scalar(0, 0, 255), 2);
        }
        utility::DrawText(img_src, faces[i].location_.tl(), text);
    }
    cv::imwrite(result_path, img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();
    return 0;
}

int TestRecognize(int argc, char *argv[]) {
    std::string result_path1 = "../../data/images/face1.jpg";
    std::string result_path2 = "../../data/images/face2.jpg";
    if (argc >= 2) {
        result_path1 = "face1.jpg";
        result_path2 = "face2.jpg";
    }

    std::cout << "Face Recognition Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = true;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    cv::Mat face1;
    cv::Mat face2;
    std::vector<cv::Point2f> keyPoints;
    ConvertKeyPoints(faces[0].keypoints_, 5, keyPoints);
    face_engine->alignFace(img_src, keyPoints, face1);
    ConvertKeyPoints(faces[1].keypoints_, 5, keyPoints);
    face_engine->alignFace(img_src, keyPoints, face2);

    std::vector<float> feature1, feature2;
    face_engine->extractFeature(face1, feature1);
    face_engine->extractFeature(face2, feature2);
    float sim = CalculateSimilarity(feature1, feature2);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        cv::Rect face = faces.at(i).location_;
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(result_path1, face1);
    cv::imwrite(result_path2, face2);
    std::cout << "similarity is: " << sim << std::endl;

    face_engine->destroyEngine();
    return 0;

}

int TestDatabase(int argc, char *argv[]) {
    std::cout << "Face Database Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);
    if (img_src.empty()) {
        std::cout << "load image failed." << std::endl;
        return 10001;
    }

    bool show_aligned = false;

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = true;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);
    face_engine->Load();
    face_engine->Clear();
    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    int faces_num = static_cast<int>(faces.size());
    std::cout << "faces number: " << faces_num << std::endl;
    for (int i = 0; i < faces_num; ++i) {
        cv::Rect face = faces.at(i).location_;

        if (show_aligned) {
#if MIRROR_BUILD_WITH_FULL_OPENCV
            cv::imshow("aligned before", img_src(face));
#endif
        }

        // align face
        cv::Mat faceAligned;
        std::vector<cv::Point2f> keyPoints;
        ConvertKeyPoints(faces.at(i).keypoints_, 5, keyPoints);
        face_engine->alignFace(img_src, keyPoints, faceAligned);

        if (show_aligned) {
#if MIRROR_BUILD_WITH_FULL_OPENCV
            cv::imshow("aligned after", faceAligned);
#endif
        }

        // get face feature
        std::vector<float> feat;
        face_engine->extractFeature(faceAligned, feat);

        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);

#if 1
        face_engine->Insert(feat, "face" + std::to_string(i));
#endif

#if 1
        QueryResult query_result;
        face_engine->QueryTop(feat, query_result);
        std::cout << i << "-th face is: " << query_result.name_ <<
                  " similarity is: " << query_result.sim_ << std::endl;
#endif

    }
    face_engine->Save();
    cv::imwrite("../../data/images/registered.jpg", img_src);

    face_engine->destroyEngine();

    return 0;
}

int TestTrack(int argc, char *argv[]) {
    std::cout << "Face Track Test......" << std::endl;

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "open camera failed." << std::endl;
        return -1;
    }

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.threadNum = 4;
    params.gpuEnabled = use_gpu;
    params.livingThreshold = 0.915;
    params.faceAntiSpoofingEnabled = use_living;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = true;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);
    face_engine->Load();

    cv::Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            continue;
        }

        double start = static_cast<double>(cv::getTickCount());

        // detect faces
        std::vector<FaceInfo> curr_faces;
        face_engine->detectFace(frame, curr_faces);

        // track faces
        std::vector<TrackedFaceInfo> faces;
        face_engine->track(curr_faces, faces);

        // anti face detect
        for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
            TrackedFaceInfo tracked_face_info = faces.at(i);
            char text[64];
            cv::Rect roi = tracked_face_info.face_info_.location_;
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);

            sprintf(text, "%s", "tracking");
            int next_y = utility::DrawText(frame, roi.tl(), text);

            bool is_living = true;
            if (params.faceAntiSpoofingEnabled) {
                float livingScore;
                is_living = face_engine->detectLivingFace(frame, roi, livingScore);
                cv::rectangle(frame, tracked_face_info.face_info_.location_,
                              cv::Scalar(0, 255, 0), 2);
                cv::Scalar color;
                if (is_living) {
                    color = cv::Scalar(0, 255, 0);
                    sprintf(text, "%s %.1f%%", "living", livingScore * 100);
                } else {
                    color = cv::Scalar(0, 0, 255);
                    sprintf(text, "%s %.1f%%", "anti living", livingScore * 100);
                }
                cv::rectangle(frame, roi, color, 2);
                next_y = utility::DrawText(frame, cv::Point2i(roi.tl().x, next_y), text);
            }

            // face recognition
            if (is_living) {
                // align face
                cv::Mat faceAligned;
                std::vector<cv::Point2f> keyPoints;
                ConvertKeyPoints(tracked_face_info.face_info_.keypoints_, 5, keyPoints);
                face_engine->alignFace(frame, keyPoints, faceAligned);

                // extract features
                std::vector<float> feat;
                face_engine->extractFeature(faceAligned, feat);

                // compare features
                QueryResult query_result;
                if (face_engine->QueryTop(feat, query_result) == 0) {
                    if (query_result.sim_ > 0.5) {
                        sprintf(text, "%s %.1f%%", query_result.name_.c_str(),
                                query_result.sim_ * 100);
                    } else {
                        sprintf(text, "%s %.1f%%", "stranger",
                                query_result.sim_ * 100);
                    }
                    utility::DrawText(frame, cv::Point2i(roi.tl().x, next_y), text);
                }
            }
        }

        double end = static_cast<double>(cv::getTickCount());
        double time_cost = (end - start) / cv::getTickFrequency();
        char text[32];
        sprintf(text, "FPS=%.2f", 1 / time_cost);

        int thickness = 1;
        float fontScale = 0.5;
        int orientation = 1; // top right
        cv::Point2i position;
        utility::GetTextCornerPosition(frame, text, orientation, fontScale, thickness, position);
        utility::DrawText(frame, position, text, fontScale, thickness);

        cv::imshow("result", frame);

        // If press space bar(32), then pause video, until any key is pressed, then restart
        if (cv::waitKey(60) == ' ') {
            cv::waitKey(-1);
        }
        if (cv::waitKey(60) == 'q' || cv::waitKey(60) == 27) {
            // press q or esc to quit,  113 is q, 27 is esc
            cv::destroyAllWindows();
            break;
        }
    }

    face_engine->destroyEngine();
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    return 0;
}

int TestFaceApi(int argc, char *argv[]) {
    std::cout << "Face API Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);
    if (img_src.empty()) {
        std::cout << "load image failed." << std::endl;
        return 10001;
    }

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceDetectorEnabled = true;
    params.faceRecognizerEnabled = true;
    params.faceAntiSpoofingEnabled = true;
    params.faceDetectorType = detectorModelType;
    params.faceRecognizerType = recognizerModelType;
    face_engine->loadModel(params);
    if (face_engine->registerFace(img_src, "test") == 0) {
        std::cout << "Register face " << "'test'" << "successfully!" << std::endl;
    } else {
        std::cout << "Register face " << "'test'" << "failed!" << std::endl;
    }
    double start = static_cast<double>(cv::getTickCount());

    bool is_living = true;
    float livingScore = 0.0f;
    VerificationResult result;
    { //  just test
        is_living = face_engine->detectLivingFace(img_src, livingScore);
        face_engine->verifyFace(img_src, result, true);
    }

    // 1. detect faces
    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    if (faces.empty()) {
        std::cout << "Cannot detect any face!" << std::endl;
        return -1;
    }

    // 2. detect living face
    FaceInfo face_info = faces[0];
    is_living = face_engine->detectLivingFace(img_src, face_info.location_, livingScore);

    // 3. verify face
    if (is_living) {
        std::cout << "Detect real face!" << std::endl;
        std::vector<cv::Point2f> keyPoints;
        ConvertKeyPoints(face_info.keypoints_, 5, keyPoints);
        int flag = face_engine->verifyFace(img_src, keyPoints, result);
        if (flag == 0) {
            if (result.sim > 0.5) {
                std::cout << "Verification successfully and similarity is : " << result.sim << std::endl;
            } else {
                std::cout << "Verification failed and similarity is : " << result.sim << std::endl;
            }
        } else {
            std::cout << "ErrorCode is :" << flag << std::endl;
        }

    } else {
        std::cout << "Detect fake face!" << std::endl;
    }

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    face_engine->destroyEngine();
}

int main(int argc, char *argv[]) {
    if (argc == 2) {
        model_path = argv[1];
    } else if (argc == 3) {
        model_path = argv[1];
        img_path = argv[2];
    } else if (argc == 4) {
        model_path = argv[1];
        img_path = argv[2];
        detectorModelType = FaceDetectorType(std::stoi(argv[3]));
    } else if (argc == 5) {
        model_path = argv[1];
        img_path = argv[2];
        detectorModelType = FaceDetectorType(std::stoi(argv[3]));
        recognizerModelType = FaceRecognizerType(std::stoi(argv[4]));
    } else if (argc >= 6) {
        model_path = argv[1];
        img_path = argv[2];
        detectorModelType = FaceDetectorType(std::stoi(argv[3]));
        recognizerModelType = FaceRecognizerType(std::stoi(argv[4]));
        use_gpu = std::string(argv[5]) == "1" ? true : false;
    }

    TestDetector(argc, argv);
    TestLandmark(argc, argv);
    TestAlignFace(argc, argv);
    TestMask(argc, argv);
    TestRecognize(argc, argv);
    TestDatabase(argc, argv);
    TestFaceApi(argc, argv);
    TestTrack(argc, argv);
    return 0;
}
